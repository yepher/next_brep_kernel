//! The per-document PARTS LIBRARY — assemblies build-spec §2.1 / §10 item 3.
//!
//! Each unique part is stored ONCE per document: `{ sourceKey, sourceSignature,
//! document (the embedded full sub-part history JSON), snapshot (the io/snapshot
//! exact-BREP payload) }`, keyed by a document-unique part name. ACOMP instance
//! features (`features/assembly_component.rs`) reference entries by name and
//! carry NO payload of their own.
//!
//! # Residency and round-trip
//!
//! The library is KERNEL-RESIDENT state (thread-local, like the history cache):
//! it is seeded through [`ingest`] (the `partsLibrary` block of a history
//! request) or [`install_parts_library`] (the app → runner channel), mutated
//! through [`add_part_to_library`] / [`refresh_library_entry`], and serialized
//! for SAVE via [`parts_library_json`] — save must serialize THIS, never echo
//! the loaded block, or healed snapshots and refreshes are lost. It clears
//! with `clear_history_cache` (document-switch semantics).
//!
//! Ingest rule (stale request blocks must never undo kernel-side state): a
//! RESIDENT entry always wins — a request echoing the block the document
//! LOADED with predates any heal/refresh, so the block only FILLS names
//! missing from the store.
//!
//! # Two seeding doors, and why they differ
//!
//! [`ingest`] is fill-only because it is fed by a *possibly stale echo*. The
//! app's history runner runs OFF the caller's thread (a native thread or a
//! browser worker), so it owns a SEPARATE store that must be kept in step with
//! the app's — and re-sending the whole block on every run is what made a
//! large assembly freeze the browser UI (each run stringified megabytes of
//! payload on the main thread). The app therefore sends the library only when
//! it CHANGES, over [`install_parts_library`], which is a *replace by content
//! identity*: it may drop and replace, not merely fill. Two guards keep that
//! safe — [`parts_library_revision`] tells the sender when to re-send, and
//! [`missing_library_parts`] lets the receiver refuse a run whose parts it
//! cannot resolve (the orphan GC below can drop entries the sender still
//! believes are resident).
//!
//! # Orphan GC
//!
//! At the end of every `execute_history` run, entries whose part name is
//! referenced by NO ACOMP feature in the request are dropped — orphaned
//! payloads never accumulate in the file. (The insert flow must therefore add
//! the ACOMP feature to the history before triggering another rebuild.)
//!
//! # Isolated sub-part execution
//!
//! [`rebuild_snapshot`] executes an embedded part document HEADLESSLY to
//! (re)produce its snapshot — the [`add_part_to_library`] insert path and the
//! ACOMP self-heal lane. It deliberately does NOT recurse into
//! `execute_history`: the history cache's `retain(request_ids)` prologue would
//! free every PARENT feature's cached handles mid-run. Instead it walks the
//! sub-document's features through `execute_feature` with a local scene,
//! brackets the scene-metadata store (the sub-part's un-namespaced records must
//! never touch the parent document's), pushes the sub-document's OWN
//! `partsLibrary` as the active library (parent and child libraries are
//! independent, spec §2.2), and frees every handle it registered before
//! returning.

use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::collections::BTreeMap;

use crate::feature_pipeline::{
    execute_feature, scene_metadata, sheet_metal, Env, FeatureDescriptor, HistoryRequest,
    PortRecord, SceneMap,
};
use crate::BrepSolid;
use wasm_bindgen::prelude::*;

// ===========================================================================
// The entry + the resident stores
// ===========================================================================

/// One unique part payload (spec §2.1). `dirty` is RESIDENT-ONLY state: set by
/// [`refresh_library_entry`] (edit-in-context / update-components), it forces
/// the ACOMP self-heal lane on the next run even with a readable snapshot, and
/// clears when the heal rewrites the snapshot.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct PartsLibraryEntry {
    #[serde(rename = "sourceKey", default)]
    pub source_key: String,
    #[serde(rename = "sourceSignature", default)]
    pub source_signature: String,
    /// The embedded full sub-part history JSON — the durable source the
    /// self-heal lane re-executes when the snapshot cache fails.
    #[serde(default)]
    pub document: serde_json::Value,
    /// The evaluated part as an `io/snapshot` payload (the fast lane).
    #[serde(default)]
    pub snapshot: String,
    /// The part's harness ports (part-local id -> record, part-local pose),
    /// captured from the same run as the snapshot. Derived like the snapshot:
    /// an entry whose document declares a PORT but carries none is healed.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub ports: BTreeMap<String, PortRecord>,
    #[serde(skip)]
    pub dirty: bool,
    /// Stable content hash of `document`, precomputed at insert/refresh/ingest
    /// so the per-feature cache hook never re-hashes a large document.
    #[serde(skip)]
    pub doc_hash: u64,
}

/// The library map shape as it travels in the history request / save file.
pub type PartsLibraryMap = BTreeMap<String, PartsLibraryEntry>;

thread_local! {
    /// The open document's library (the ROOT store).
    static ROOT: RefCell<PartsLibraryMap> = RefCell::new(BTreeMap::new());
    /// Active-library stack for isolated sub-document runs: a nested ACOMP
    /// resolves against ITS document's library, never the parent's.
    static STACK: RefCell<Vec<PartsLibraryMap>> = const { RefCell::new(Vec::new()) };
    /// Monotonic REVISION of the ROOT store, bumped by every mutation that
    /// actually CHANGES it (insert, refresh, heal, GC drop, install, clear).
    /// The app polls it to decide whether a background runner's resident copy
    /// is stale — the library no longer rides on every history request, so
    /// this counter is what tells the app when to re-send it. Thread-local
    /// like the store itself: main and each runner count their own store.
    static REVISION: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Record that the ROOT store changed. Call it from EVERY mutation that
/// actually alters content — never speculatively: `ingest` and
/// `gc_after_rebuild` run on every history run, so an unconditional bump there
/// would make the app re-send the whole library every edit, which is the cost
/// this counter exists to avoid.
fn bump_revision() {
    REVISION.with(|revision| revision.set(revision.get().wrapping_add(1)));
}

/// The ROOT store's current revision (see [`REVISION`]). Two reads that differ
/// mean the library changed in between; two that agree mean it did not.
pub fn parts_library_revision() -> u64 {
    REVISION.with(std::cell::Cell::get)
}

/// Stable content hash of a JSON value: sorted-key walk (never trusts a
/// serializer's map ordering), string leaves verbatim.
pub fn stable_json_hash(value: &serde_json::Value) -> u64 {
    use std::hash::{Hash, Hasher};
    fn walk(value: &serde_json::Value, hasher: &mut impl Hasher) {
        match value {
            serde_json::Value::Null => 0u8.hash(hasher),
            serde_json::Value::Bool(flag) => {
                1u8.hash(hasher);
                flag.hash(hasher);
            }
            serde_json::Value::Number(number) => {
                2u8.hash(hasher);
                number.to_string().hash(hasher);
            }
            serde_json::Value::String(text) => {
                3u8.hash(hasher);
                text.hash(hasher);
            }
            serde_json::Value::Array(items) => {
                4u8.hash(hasher);
                for item in items {
                    walk(item, hasher);
                }
            }
            serde_json::Value::Object(map) => {
                5u8.hash(hasher);
                let mut keys: Vec<&String> = map.keys().collect();
                keys.sort();
                for key in keys {
                    key.hash(hasher);
                    walk(&map[key], hasher);
                }
            }
        }
    }
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    walk(value, &mut hasher);
    hasher.finish()
}

/// Read the entry for a part name from the ACTIVE library: the innermost
/// isolated-run library when one is pushed, else the root store.
pub(crate) fn active_entry(part_name: &str) -> Option<PartsLibraryEntry> {
    STACK.with(|stack| {
        let stack = stack.borrow();
        if let Some(top) = stack.last() {
            return top.get(part_name).cloned();
        }
        ROOT.with(|root| root.borrow().get(part_name).cloned())
    })
}

/// Write a healed snapshot back into the ACTIVE library entry and clear its
/// dirty flag. A heal inside an isolated run updates the pushed (temporary)
/// library only — the parent entry's embedded document is NOT rewritten (its
/// re-snapshot bakes the healed geometry anyway; the stale inner snapshot heals
/// again on the next full sub-document run).
pub(crate) fn heal_entry(part_name: &str, snapshot: String, ports: BTreeMap<String, PortRecord>) {
    let heal = |map: &mut PartsLibraryMap| {
        if let Some(entry) = map.get_mut(part_name) {
            entry.snapshot = snapshot.clone();
            entry.ports = ports.clone();
            entry.dirty = false;
        }
    };
    STACK.with(|stack| {
        let mut stack = stack.borrow_mut();
        if let Some(top) = stack.last_mut() {
            heal(top);
        } else {
            ROOT.with(|root| heal(&mut root.borrow_mut()));
            // A heal rewrites a ROOT entry's snapshot, so the store changed.
            // (It does NOT change the entry's content IDENTITY — see
            // `install` — so re-sending a healed library never clobbers a
            // runner's own, independently-derived heal.)
            bump_revision();
        }
    });
}

/// Drop the whole library (document switch — wired into `clear_history_cache`).
pub(crate) fn clear_all() {
    let had_entries = ROOT.with(|root| {
        let mut root = root.borrow_mut();
        let had = !root.is_empty();
        root.clear();
        had
    });
    STACK.with(|stack| stack.borrow_mut().clear());
    if had_entries {
        bump_revision();
    }
}

// ===========================================================================
// execute_history hooks: ingest, cache fingerprint, GC
// ===========================================================================

/// The ACOMP dispatch predicate — shared by the fingerprint hook, the GC scan,
/// and (as adjacent literals) the `execute_feature` match arm, so they can
/// never disagree on what counts as a component instance.
pub(crate) fn is_acomp_type(feature_type: &str) -> bool {
    matches!(feature_type, "ACOMP" | "ASSEMBLY COMPONENT")
}

/// Merge a request's `partsLibrary` block into the root store (start of every
/// `execute_history` run — this is how a LOADED document seeds the library).
/// A RESIDENT entry always wins: mid-session the kernel store is the source of
/// truth (it carries heals, refreshes, and pending-dirty state the request's
/// echo of the loaded block predates); the block only ever FILLS missing names
/// (document load starts from an empty store, cleared by `clear_history_cache`).
pub(crate) fn ingest(request_map: &PartsLibraryMap) {
    if request_map.is_empty() {
        return;
    }
    let seeded = ROOT.with(|root| {
        let mut root = root.borrow_mut();
        let mut seeded = false;
        for (name, incoming) in request_map {
            if root.contains_key(name) {
                continue;
            }
            let mut entry = incoming.clone();
            // A loaded block's entry gets the same stamp as an inserted one, so a
            // document saved before per-loop ids existed picks them up on open.
            stamp_document_loop_ids(&mut entry.document);
            entry.doc_hash = stable_json_hash(&entry.document);
            // A document saved before parts carried their ports heals once.
            entry.dirty = needs_port_heal(&entry);
            root.insert(name.clone(), entry);
            seeded = true;
        }
        seeded
    });
    if seeded {
        bump_revision();
    }
}

/// Whether an entry must heal to pick up its ports: its document declares a
/// PORT feature (at any depth — a nested assembly's parts count) but the entry
/// carries no port records, i.e. it was saved before parts carried them.
fn needs_port_heal(entry: &PartsLibraryEntry) -> bool {
    entry.ports.is_empty() && document_declares_ports(&entry.document)
}

/// Whether any feature in `document` — or in a library entry it embeds — is
/// a PORT.
fn document_declares_ports(document: &serde_json::Value) -> bool {
    match document {
        serde_json::Value::Object(map) => {
            map.get("type").and_then(serde_json::Value::as_str) == Some("PORT")
                || map.values().any(document_declares_ports)
        }
        serde_json::Value::Array(items) => items.iter().any(document_declares_ports),
        _ => false,
    }
}

/// The CONTENT IDENTITY of an entry: what makes two entries the same PART.
/// `snapshot` is deliberately excluded — a heal re-derives the snapshot from
/// the same `document`, so it must not read as a different part. This is the
/// same identity [`mix_descriptor_fingerprint`] mixes into an ACOMP's cache
/// fingerprint, so "install kept the resident entry" and "the instance
/// replayed from cache" can never disagree.
fn identity(entry: &PartsLibraryEntry) -> (&str, &str, u64) {
    (&entry.source_key, &entry.source_signature, entry.doc_hash)
}

/// INSTALL a library wholesale — the app → history-runner channel (the runner
/// owns its own thread-local store, and the per-run request no longer carries
/// the block, so this is how a background thread/worker learns the library).
///
/// Unlike [`ingest`] (fill-only, resident-wins) this makes the store MATCH
/// `incoming` BY CONTENT IDENTITY, which is what a cache channel needs:
///
/// * a name missing from the store is inserted (stamped + hashed like ingest);
/// * a name whose resident entry has the SAME identity is KEPT VERBATIM — that
///   is what protects a runner-side heal, whose only trace is a rewritten
///   `snapshot`, from being clobbered by a re-send;
/// * a name whose resident entry has a DIFFERENT identity is REPLACED and
///   marked `dirty`, so the ACOMP self-heal lane re-derives every instance
///   (`dirty` is `#[serde(skip)]` and cannot ride the wire — it is derived
///   HERE, from the identity mismatch);
/// * a resident name ABSENT from `incoming` is dropped.
///
/// Returns whether anything changed (and bumps the revision if so).
pub fn install_parts_library(incoming: &PartsLibraryMap) -> bool {
    let mut changed = false;
    ROOT.with(|root| {
        let mut root = root.borrow_mut();
        root.retain(|name, _| {
            let keep = incoming.contains_key(name);
            changed |= !keep;
            keep
        });
        for (name, entry) in incoming {
            let mut entry = entry.clone();
            stamp_document_loop_ids(&mut entry.document);
            entry.doc_hash = stable_json_hash(&entry.document);
            entry.dirty = needs_port_heal(&entry);
            match root.get(name) {
                // Same part: keep the RESIDENT copy — it may carry a heal this
                // side derived and the sender never saw.
                Some(resident) if identity(resident) == identity(&entry) => continue,
                // Genuinely different content: replace, and force the self-heal
                // lane so every instance follows the new document.
                Some(_) => entry.dirty = true,
                None => {}
            }
            root.insert(name.clone(), entry);
            changed = true;
        }
    });
    if changed {
        bump_revision();
    }
    changed
}

/// A CLONE of the root store — what the app hands a runner through
/// [`install_parts_library`]. (`parts_library_json` is the SAVE door; this is
/// the in-process one, with no JSON round trip on the native paths.)
pub fn parts_library_map() -> PartsLibraryMap {
    ROOT.with(|root| root.borrow().clone())
}

/// The PREFLIGHT for a run: every part name the request's ACOMP features
/// reference that the root store cannot resolve, in request order.
///
/// A background runner calls this BEFORE executing a run and refuses the run
/// when it is non-empty. That is the drift-proof half of the library channel:
/// the orphan GC at the end of every run can legitimately drop entries the
/// sender still believes are resident (undo to zero components GCs the runner's
/// store while the sender's store, which never ran, keeps them), and no
/// revision bookkeeping can see that. Missing CONTENT is observable; that is
/// what this checks.
pub fn missing_library_parts(request: &HistoryRequest) -> Vec<String> {
    let mut missing = Vec::new();
    ROOT.with(|root| {
        let root = root.borrow();
        for descriptor in &request.features {
            if !is_acomp_type(&descriptor.feature_type) {
                continue;
            }
            let Some(name) = descriptor
                .input_params
                .get("partName")
                .and_then(|value| value.as_str())
                .map(str::trim)
                .filter(|name| !name.is_empty())
            else {
                continue;
            };
            if !root.contains_key(name) && !missing.iter().any(|seen| seen == name) {
                missing.push(name.to_string());
            }
        }
    });
    missing
}

/// The history-cache hook for one descriptor: mix the referenced library
/// entry's CONTENT identity (source key/signature/document hash — snapshot
/// deliberately excluded, a heal must not dirty the instance) into the
/// feature fingerprint, and report whether a DIRTY entry forces re-execution.
/// Non-ACOMP descriptors pass through untouched.
pub(crate) fn mix_descriptor_fingerprint(
    descriptor: &FeatureDescriptor,
    fingerprint: u64,
) -> (u64, bool) {
    if !is_acomp_type(&descriptor.feature_type) {
        return (fingerprint, false);
    }
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    fingerprint.hash(&mut hasher);
    let part_name = descriptor
        .input_params
        .get("partName")
        .and_then(|value| value.as_str())
        .unwrap_or("");
    part_name.hash(&mut hasher);
    let mut force_dirty = false;
    match active_entry(part_name) {
        Some(entry) => {
            1u8.hash(&mut hasher);
            entry.source_key.hash(&mut hasher);
            entry.source_signature.hash(&mut hasher);
            entry.doc_hash.hash(&mut hasher);
            force_dirty = entry.dirty;
        }
        None => 0u8.hash(&mut hasher),
    }
    (hasher.finish(), force_dirty)
}

/// Orphan GC (end of every `execute_history` run): retain only entries some
/// ACOMP feature in the REQUEST references (the request always carries the full
/// feature list — editor stop points truncate execution, not the list).
pub(crate) fn gc_after_rebuild(request: &HistoryRequest) {
    let referenced: std::collections::BTreeSet<String> = request
        .features
        .iter()
        .filter(|descriptor| is_acomp_type(&descriptor.feature_type))
        .filter_map(|descriptor| {
            descriptor
                .input_params
                .get("partName")
                .and_then(|value| value.as_str())
                .map(|name| name.trim().to_string())
        })
        .collect();
    let dropped = ROOT.with(|root| {
        let mut root = root.borrow_mut();
        let before = root.len();
        root.retain(|name, _| referenced.contains(name));
        before != root.len()
    });
    if dropped {
        bump_revision();
    }
}

// ===========================================================================
// Isolated sub-part execution (the self-heal / insert lane)
// ===========================================================================

/// Execute an embedded part document headlessly and return its surviving
/// solids as PART-LOCAL `(scene name, owned BREP)` pairs in history order.
/// Never touches the history cache or the parent scene; every handle it
/// registers is freed before returning. See the module doc for why this must
/// not recurse into `execute_history`.
fn run_isolated_document(document: &serde_json::Value) -> Result<IsolatedRun, String> {
    let mut request: HistoryRequest = serde_json::from_value(document.clone())
        .map_err(|error| format!("embedded part document does not parse: {error}"))?;
    // A stale editor stop point saved into a library part must not silently
    // truncate the part on every heal.
    request.stop_at_id = None;
    request.stop_before_id = None;

    // The sub-document's own library is the active one for this run (a nested
    // assembly's ACOMPs resolve against it; missing entries are clean errors,
    // never a fallback to the parent's library). Its entries arrive raw from
    // the JSON, so the port heal is decided here as `ingest` decides it for a
    // loaded document — an inner part saved before parts carried their ports
    // must not take the fast lane and publish none.
    STACK.with(|stack| {
        let mut library = std::mem::take(&mut request.parts_library);
        for entry in library.values_mut() {
            entry.dirty = needs_port_heal(entry);
        }
        stack.borrow_mut().push(library);
    });
    let outcome = run_isolated_features(&request);
    STACK.with(|stack| {
        stack.borrow_mut().pop();
    });
    outcome
}

/// The feature walk of [`run_isolated_document`], separated so the library
/// stack push/pop brackets every exit path.
/// What an isolated part run yields: the surviving solids (part-local names,
/// history order) and the part's ports (part-local ids, part-local pose).
type IsolatedRun = (Vec<(String, BrepSolid)>, BTreeMap<String, PortRecord>);

fn run_isolated_features(request: &HistoryRequest) -> Result<IsolatedRun, String> {
    let env = Env::build(&request.expressions, &request.configurator).unwrap_or_else(Env::poisoned);
    let mut scene = SceneMap::default();
    let mut results = Vec::with_capacity(request.features.len());
    let mut all_handles: Vec<u32> = Vec::new();
    let mut error: Option<String> = None;
    for descriptor in &request.features {
        let result = execute_feature(descriptor, &env, &scene);
        scene.apply(&result);
        all_handles.extend(result.added.iter().map(|added| added.handle));
        // Replicate the loop's producer stamp so the snapshot carries the same
        // topology metadata a real history run would have written.
        if !result.id.is_empty() {
            let mut seed = serde_json::Map::new();
            seed.insert(
                "sourceFeatureId".into(),
                serde_json::Value::String(result.id.clone()),
            );
            for added in &result.added {
                for (_, face_name) in &added.face_names {
                    scene_metadata::merge_record(face_name, &seed, false);
                }
                for (_, edge_name) in &added.edge_names {
                    scene_metadata::merge_record(edge_name, &seed, false);
                }
            }
        }
        if let Some(message) = &result.error {
            error = Some(format!("feature '{}': {message}", result.id));
            break;
        }
        results.push(result);
    }

    // Survivors: added solids still scene-resident under their own name, in
    // history order (a consumed/superseded solid's name no longer maps to it).
    let mut members = Vec::new();
    if error.is_none() {
        for result in &results {
            for added in &result.added {
                if scene.resolve_solid(&added.name) == Some(added.handle) {
                    let solid =
                        crate::with_registered_solid_str(added.handle, |solid| Ok(solid.clone()))?;
                    members.push((added.name.clone(), solid));
                }
            }
        }
    }
    for handle in all_handles {
        sheet_metal::remove_tree(handle);
        crate::free_registered_solid(handle);
    }
    // The part's ports: every record the run left in the scene (a nested
    // assembly's component ports are already namespaced by its ACOMPs).
    let ports: BTreeMap<String, PortRecord> = scene
        .ports
        .iter()
        .map(|(id, port)| (id.clone(), port.clone()))
        .collect();
    match error {
        Some(message) => Err(message),
        None if members.is_empty() => Err("embedded part document produced no solids".into()),
        None => Ok((members, ports)),
    }
}

/// Execute an embedded part document and snapshot the result — the ONE
/// snapshot-(re)build lane ([`add_part_to_library`] insert + ACOMP self-heal).
/// The scene-metadata store is bracketed around the run: the sub-part's
/// un-namespaced records feed the snapshot capture and are then discarded, so
/// the parent document's store is byte-identical afterwards on every path.
pub(crate) fn rebuild_snapshot(
    document: &serde_json::Value,
) -> Result<(String, BTreeMap<String, PortRecord>), String> {
    let saved = scene_metadata::take_store();
    let outcome = run_isolated_document(document).and_then(|(members, ports)| {
        let refs: Vec<(&str, &BrepSolid)> = members
            .iter()
            .map(|(name, solid)| (name.as_str(), solid))
            .collect();
        crate::snapshot_solids(&refs).map(|snapshot| (snapshot, ports))
    });
    scene_metadata::restore_store(saved);
    outcome
}

// ===========================================================================
// The app-facing insert / update / save surface
// ===========================================================================

/// A part name not yet used in the root store: `requested`, else
/// `requested-2`, `requested-3`, …
fn unique_name(root: &PartsLibraryMap, requested: &str) -> String {
    if !root.contains_key(requested) {
        return requested.to_string();
    }
    for counter in 2.. {
        let candidate = format!("{requested}-{counter}");
        if !root.contains_key(&candidate) {
            return candidate;
        }
    }
    unreachable!("counter loop is unbounded")
}

/// Stamp per-loop sketch ids onto an entry's document, in place — run at EVERY
/// door into the library (insert, refresh, and the loaded-block seeding), so a
/// part gets durable loop ids no matter how it arrived.
///
/// The engine also stamps a document on LOAD
/// (`EngineState::set_history_json`), but a part inserted or refreshed through
/// the assembly lanes never passes through that: `add_part_to_library` and
/// `refresh_library_entry` take a document STRING straight from a caller (the
/// file panel inserts a saved `.BREP.json`; update-components commits an edited
/// one). Stamping here rather than at those call sites means a future lane that
/// installs a document cannot silently skip it.
///
/// This renames nothing — the kernel derives the same loop ids every run. What
/// it adds is durability: a stored id survives deleting the edge it was derived
/// from, which derivation alone cannot. See `features/sketch/loop_ids`.
fn stamp_document_loop_ids(document: &mut serde_json::Value) {
    let Some(features) = document
        .get_mut("features")
        .and_then(serde_json::Value::as_array_mut)
    else {
        return;
    };
    for feature in features {
        if feature.get("type").and_then(serde_json::Value::as_str) != Some("S") {
            continue;
        }
        let Some(sketch) = feature
            .get_mut("persistentData")
            .and_then(|data| data.get_mut("sketch"))
        else {
            continue;
        };
        crate::feature_pipeline::assign_sketch_loop_ids(sketch);
    }
}

/// INSERT: hand the opened sub-part document over to the library. Executes the
/// document once, snapshots, stores the entry, and returns the EFFECTIVE part
/// name (the ACOMP `partName` the app must reference):
///
/// - a live entry with the same `sourceKey` AND `sourceSignature` is reused
///   verbatim (no re-execution — instant re-insert, spec §2.1);
/// - same `sourceKey` but a DIFFERENT signature gets a fresh entry under a
///   disambiguated name (existing instances keep their version; the explicit
///   refresh lane is [`refresh_library_entry`]);
/// - a requested name taken by different content is disambiguated (`name-2`).
#[wasm_bindgen]
pub fn add_part_to_library(
    name: &str,
    source_key: &str,
    source_signature: &str,
    document_json: &str,
) -> Result<String, JsValue> {
    let requested = name.trim();
    if requested.is_empty() {
        return Err(JsValue::from_str("add_part_to_library: empty part name"));
    }
    let reused = ROOT.with(|root| {
        root.borrow().iter().find_map(|(entry_name, entry)| {
            (entry.source_key == source_key && entry.source_signature == source_signature)
                .then(|| entry_name.clone())
        })
    });
    if let Some(entry_name) = reused {
        return Ok(entry_name);
    }
    let mut document: serde_json::Value = serde_json::from_str(document_json)
        .map_err(|error| JsValue::from_str(&format!("add_part_to_library: {error}")))?;
    // Stamp BEFORE the hash and the snapshot so both describe the stored form —
    // a later re-stamp then finds nothing to change and cannot dirty the entry.
    stamp_document_loop_ids(&mut document);
    let (snapshot, ports) = rebuild_snapshot(&document)
        .map_err(|error| JsValue::from_str(&format!("add_part_to_library: {error}")))?;
    let entry = PartsLibraryEntry {
        source_key: source_key.to_string(),
        source_signature: source_signature.to_string(),
        doc_hash: stable_json_hash(&document),
        document,
        snapshot,
        ports,
        dirty: false,
    };
    let final_name = ROOT.with(|root| {
        let mut root = root.borrow_mut();
        let final_name = unique_name(&root, requested);
        root.insert(final_name.clone(), entry);
        final_name
    });
    bump_revision();
    Ok(final_name)
}

/// UPDATE (update-components / edit-in-place commit): replace an existing
/// entry's document + signature and mark it DIRTY. The next history run's
/// ACOMP self-heal lane re-executes the document, re-snapshots, and every
/// instance of the part follows.
#[wasm_bindgen]
pub fn refresh_library_entry(
    name: &str,
    source_signature: &str,
    document_json: &str,
) -> Result<(), JsValue> {
    let mut document: serde_json::Value = serde_json::from_str(document_json)
        .map_err(|error| JsValue::from_str(&format!("refresh_library_entry: {error}")))?;
    // Stamped before the hash, as in `add_part_to_library`.
    stamp_document_loop_ids(&mut document);
    ROOT.with(|root| {
        let mut root = root.borrow_mut();
        let Some(entry) = root.get_mut(name) else {
            return Err(JsValue::from_str(&format!(
                "refresh_library_entry: no parts-library entry named '{name}'"
            )));
        };
        entry.source_signature = source_signature.to_string();
        entry.doc_hash = stable_json_hash(&document);
        entry.document = document;
        entry.ports.clear(); // the heal re-derives them from the new document
        entry.dirty = true;
        Ok(())
    })?;
    bump_revision();
    Ok(())
}

/// The current library in the `partsLibrary` block shape (spec §2.1) — what
/// SAVE must serialize into the document (never the loaded block: this copy
/// carries healed snapshots, refreshes, and GC).
#[wasm_bindgen]
pub fn parts_library_json() -> String {
    ROOT.with(|root| {
        serde_json::to_string(&*root.borrow()).unwrap_or_else(|_| "{}".to_string())
    })
}

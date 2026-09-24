use crate::boolean::{assemble_open_fragments, edge_interior_lies_on, finalize_assembled_solid};
use crate::classification::{parameter_point_in_face, PolygonClass};
use crate::face_merge::merge_same_surface_faces_open;
use crate::fragment::{fragment_solid, FaceFragmentRecord, FragmentEdgeSource};
use crate::imprint::{
    EdgeSplitRecord, FaceImprints, FaceKey, FacePcurve, ImprintOptions, ImprintPieceRecord,
    ImprintResultRecord, ImprintVertex,
};
use crate::mass_properties::parameter_space_area;
use crate::project_point_to_surface;
use crate::topology::{
    BrepSolid, CoedgeRecord, EdgeRecord, FaceRecord, LoopRecord, ShellRecord, VertexRecord,
};
use crate::{
    apply_edge_splits, build_imprints, classify_point, offset_face_carrier, DiagnosticSeverity,
    KernelDiagnostics, KernelOutcome, KernelStage, KernelTolerances, PointClass, Vec2, Vec3,
};
use rustc_hash::{FxHashMap as HashMap, FxHashSet as HashSet};
use serde::Serialize;

const SOURCE_OPERAND: u8 = 250;

fn debug_enabled() -> bool {
    std::env::var("BREP_OS_DEBUG").is_ok_and(|value| !value.is_empty() && value != "0")
}

macro_rules! os_debug {
    ($($arg:tt)*) => {
        if debug_enabled() {
            eprintln!($($arg)*);
        }
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum OffsetFaceRole {
    Source,
    Offset,
    Wall,
}

#[derive(Clone, Debug, Serialize)]
pub struct OffsetShellResultRecord {
    pub solid: BrepSolid,
    pub face_images: Vec<OffsetShellFaceImageRecord>,
}

#[derive(Clone, Debug, Serialize)]
pub struct OffsetShellFaceImageRecord {
    pub role: OffsetFaceRole,
    pub source_face_id: u64,
}

#[derive(Clone)]
struct Carrier {
    solid: BrepSolid,
    source_face_id: u64,
    kind: OffsetFaceRole,
}

#[path = "offset_shell/carriers.rs"]
mod carriers;
#[path = "offset_shell/smooth_sync.rs"]
mod smooth_sync;
#[path = "offset_shell/carrier_rebuild.rs"]
mod carrier_rebuild;
#[path = "offset_shell/connectors.rs"]
mod connectors;
#[path = "offset_shell/orientation.rs"]
mod orientation;
#[path = "offset_shell/rim_welds.rs"]
mod rim_welds;
#[path = "offset_shell/pipeline.rs"]
mod pipeline;
// BREP private tests: 77b8029939f98de3

use carrier_rebuild::*;
use carriers::*;
use connectors::*;
use orientation::*;
use rim_welds::*;
use smooth_sync::*;

pub use pipeline::{offset_shell, offset_shell_with_diagnostics};
pub(crate) use orientation::{flip_all_faces, flip_shell_faces, orient_open_solid_faces};

/// Merge connected shells, preserving the first shell record and face encounter order.
fn merge_connected_shells(solid: &mut BrepSolid, shell_unions: Vec<(usize, usize)>) {
    if shell_unions.is_empty() {
        return;
    }
    let mut parent = (0..solid.shells.len()).collect::<Vec<_>>();
    fn root(parent: &mut [usize], index: usize) -> usize {
        if parent[index] != index {
            parent[index] = root(parent, parent[index]);
        }
        parent[index]
    }
    for (first, second) in shell_unions {
        let first_root = root(&mut parent, first);
        let second_root = root(&mut parent, second);
        if first_root != second_root {
            parent[second_root] = first_root;
        }
    }
    let original = std::mem::take(&mut solid.shells);
    let mut merged = Vec::<ShellRecord>::new();
    let mut group_of = HashMap::<usize, usize>::default();
    for (index, shell) in original.into_iter().enumerate() {
        let group = root(&mut parent, index);
        if let Some(target) = group_of.get(&group).copied() {
            merged[target].faces.extend(shell.faces);
        } else {
            group_of.insert(group, merged.len());
            merged.push(shell);
        }
    }
    solid.shells = merged;
}

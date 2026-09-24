use crate::arrangement::Vec2;
use crate::spatial::{Aabb, Bvh};
use crate::topology::{BrepSolid, FaceRecord};
use crate::{intersect_curve_surface, make_line, project_point_to_surface, NurbsCurve, Vec3};
use serde::Serialize;

#[path = "classification/face_containment.rs"]
mod face_containment;
#[path = "classification/solid_classifier.rs"]
mod solid_classifier;
// BREP private tests: 31c1a4888bd0d619

pub use face_containment::{parameter_point_in_face, PolygonClass};
pub use solid_classifier::{classify_point, PointClass, PointClassification, SolidClassifier};

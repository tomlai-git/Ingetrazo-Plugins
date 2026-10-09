# SPDX-License-Identifier: GPL-3.0-or-later
"""
IngeTrazo Unfold — Classic workflow for v0.5.7

Workflow:
  1. Extensions -> Unfold -> Unfold
  2. DOUBLE-CLICK a face to select all coplanar faces.
  3. Invoke Unfold again.
  4. SINGLE-CLICK the adjacent reference face.

The selected sheet is rotated around its actual shared hinge.
The unfolded faces are rebuilt separately from surrounding geometry.
Orphan edges left by removing the original faces are deleted.

The hinge remains visually joined to the reference face.
"""

from __future__ import annotations

import copy
import logging
import math

from PySide6.QtCore import QObject, QEvent, Qt
from PySide6.QtGui import QVector3D

from core.history import Command
from core.mesh import Face

LOG = logging.getLogger("ingetrazo.plugins.unfold_classic")

EPS = 1e-7
PLANE_TOL = 1e-5
NORMAL_TOL = 1e-5


def _q(p):
    return QVector3D(p)


def _face_plane(face):
    """Return a unit normal and plane offset."""
    n = _q(face.normal())

    if n.length() < EPS:
        return None

    n.normalize()
    p = _q(face.loop[0].position)

    return n, QVector3D.dotProduct(n, p)


def _same_plane(a, b):
    """Check geometric coplanarity, regardless of normal direction."""
    pa = _face_plane(a)
    pb = _face_plane(b)

    if pa is None or pb is None:
        return False

    na, da = pa
    nb, db = pb

    dot = QVector3D.dotProduct(na, nb)

    if abs(abs(dot) - 1.0) > NORMAL_TOL:
        return False

    if dot < 0:
        db = -db

    return abs(da - db) <= PLANE_TOL


def _coplanar_faces(scene, ref):
    return [
        face for face in scene.mesh.faces
        if isinstance(face, Face)
        and scene.entity_selectable(face)
        and _same_plane(ref, face)
    ]


def _axis_rotation_point(p, a, axis, angle):
    """Rotate a point about the line a + t*axis."""
    v = _q(p) - _q(a)
    u = _q(axis)

    length = u.length()

    if length < EPS:
        return _q(p)

    u = u / length

    c = math.cos(angle)
    s = math.sin(angle)

    return (
        _q(a)
        + v * c
        + QVector3D.crossProduct(u, v) * s
        + u * QVector3D.dotProduct(u, v) * (1.0 - c)
    )


def _rotate_normal(n, axis, angle):
    return _axis_rotation_point(
        n, QVector3D(0, 0, 0), axis, angle
    )


def _plane_intersection(p1, p2):
    """Return a point and direction for two intersecting planes."""
    n1, d1 = p1
    n2, d2 = p2

    direction = QVector3D.crossProduct(n1, n2)
    denom = QVector3D.dotProduct(direction, direction)

    if denom < EPS:
        return None

    point = QVector3D.crossProduct(
        n1 * d2 - n2 * d1,
        direction,
    ) / denom

    direction.normalize()

    return point, direction


def _rotation_to_reference(ref, target):
    """Calculate the rotation taking target's normal to ref's normal."""
    pref = _face_plane(ref)
    ptar = _face_plane(target)

    if pref is None or ptar is None:
        return None

    n_ref, _ = pref
    n_tar, _ = ptar

    line = _plane_intersection(pref, ptar)

    if line is None:
        return None

    point, axis = line

    cross = QVector3D.crossProduct(n_tar, n_ref)

    sinv = QVector3D.dotProduct(cross, axis)
    cosv = max(
        -1.0,
        min(1.0, QVector3D.dotProduct(n_tar, n_ref)),
    )

    angle = math.atan2(sinv, cosv)

    test = _rotate_normal(n_tar, axis, angle)

    if (test - n_ref).length() > 1e-4:
        angle = -angle

    return point, axis, angle


def _copy_face_data(face):
    return copy.deepcopy(face.attrs or {})


def _selected_faces(scene):
    return [
        item for item in scene.selection
        if isinstance(item, Face)
        and item in scene.mesh.faces
    ]


def _delete_orphan_edges(mesh):
    """
    Delete edges with no attached faces.

    Run this after removing the original target faces and before rebuilding
    the unfolded sheet. Edges still used by the reference or surrounding
    geometry have attached faces and are retained.
    """
    removed = 0

    for edge in list(mesh.edges):
        if not edge.faces:
            mesh.remove_edge(edge)
            removed += 1

    return removed


class UnfoldCommand(Command):
    """Single undoable operation, including geometry cleanup."""

    def __init__(self, mutate):
        self.mutate = mutate
        self.before = None
        self.after = None

    @staticmethod
    def _capture(scene):
        return {
            "mesh": scene.mesh.capture_state(),
            "selection": set(scene.selection),
            "version": scene.version,
        }

    @staticmethod
    def _restore(scene, state):
        scene.mesh.restore_state(state["mesh"])

        valid = (
            set(scene.mesh.faces)
            | set(scene.mesh.edges)
            | set(scene.groups)
        )

        scene.selection.clear()
        scene.selection.update(
            item for item in state["selection"]
            if item in valid
        )

    def do(self, scene):
        if self.after is None:
            self.before = self._capture(scene)

            try:
                self.mutate(scene)
            except BaseException:
                self._restore(scene, self.before)
                raise

            self.after = self._capture(scene)
        else:
            self._restore(scene, self.after)

        scene.version += 1

    def undo(self, scene):
        self._restore(scene, self.before)
        scene.version += 1


def _unfold_to_reference(viewport, ref):
    """Unfold selected faces to the clicked reference face."""
    from core.history import rotation_matrix

    scene = viewport.scene
    mesh = scene.mesh

    targets = [
        face for face in _selected_faces(scene)
        if face is not ref and face in mesh.faces
    ]

    if not targets:
        viewport.flash_status(
            "Unfold: no selected faces to unfold.", 4000
        )
        return False

    target_set = set(targets)
    hinge = None
    hinge_face = None

    # Find the actual shared edge between reference and selected group.
    for loop in (ref.loop, *ref.hole_loops):
        for i, va in enumerate(loop):
            vb = loop[(i + 1) % len(loop)]
            edge = mesh.find_edge(va, vb)

            if edge is None:
                continue

            touching = [
                face for face in edge.faces
                if face in target_set
            ]

            if touching:
                hinge = edge
                hinge_face = touching[0]
                break

        if hinge is not None:
            break

    if hinge is None or hinge_face is None:
        viewport.flash_status(
            "Unfold: reference face must share an actual edge "
            "with the selected group.",
            5000,
        )
        return False

    axis_a = QVector3D(hinge.v0.position)
    axis_b = QVector3D(hinge.v1.position)
    axis = axis_b - axis_a

    if axis.length() < EPS:
        viewport.flash_status(
            "Unfold: common hinge edge is degenerate.", 4000
        )
        return False

    axis.normalize()

    n_ref = QVector3D(ref.normal())
    n_tar = QVector3D(hinge_face.normal())

    if n_ref.length() < EPS or n_tar.length() < EPS:
        viewport.flash_status(
            "Unfold: could not determine face normals.", 4000
        )
        return False

    n_ref.normalize()
    n_tar.normalize()

    angle_deg = math.degrees(math.atan2(
        QVector3D.dotProduct(
            QVector3D.crossProduct(n_tar, n_ref), axis
        ),
        max(
            -1.0,
            min(1.0, QVector3D.dotProduct(n_tar, n_ref))
        ),
    ))

    # Confirm the sign convention used by IngeTrazo's rotation matrix.
    test_matrix = rotation_matrix(
        QVector3D(0.0, 0.0, 0.0),
        axis,
        angle_deg,
    )

    test_normal = test_matrix.mapVector(n_tar)

    if test_normal.length() > EPS:
        test_normal.normalize()

    if (test_normal - n_ref).length() > 1e-4:
        angle_deg = -angle_deg

    if abs(angle_deg) < 1e-7:
        viewport.flash_status(
            "Unfold: selected sheet is already aligned "
            "with the reference plane.",
            3500,
        )
        return False

    angle = math.radians(angle_deg)

    # Store the complete original geometry and attributes.
    face_specs = []

    for face in targets:
        face_specs.append({
            "attrs": _copy_face_data(face),
            "loop": [
                QVector3D(v.position) for v in face.loop
            ],
            "holes": [
                [QVector3D(v.position) for v in hole]
                for hole in face.hole_loops
            ],
        })

    def mutate(current_scene):
        current_mesh = current_scene.mesh

        # Remove the original selected faces.
        for face in targets:
            if face in current_mesh.faces:
                current_mesh.remove_face(face)

        # NEW: delete edges orphaned by removal of the original faces.
        # Shared hinge edges are retained because the reference face
        # still uses them.
        orphan_count = _delete_orphan_edges(current_mesh)

        rebuilt = []

        # Recreate all selected faces at their rotated positions.
        # add_face welds coincident positions within the rebuilt sheet,
        # while the old surrounding geometry remains in place.
        for spec in face_specs:
            rotated_loop = [
                _axis_rotation_point(p, axis_a, axis, angle)
                for p in spec["loop"]
            ]

            rotated_holes = [
                [
                    _axis_rotation_point(p, axis_a, axis, angle)
                    for p in hole
                ]
                for hole in spec["holes"]
            ]

            new_face = current_mesh.add_face(
                rotated_loop,
                rotated_holes,
            )

            if new_face is None:
                raise RuntimeError(
                    "IngeTrazo could not rebuild an unfolded face."
                )

            new_face.attrs = copy.deepcopy(spec["attrs"])
            rebuilt.append(new_face)

        current_scene.selection.clear()
        current_scene.selection.update(rebuilt)

        LOG.info(
            "Unfold rebuilt %d faces; removed %d orphan edges.",
            len(rebuilt),
            orphan_count,
        )

    try:
        viewport.history.execute(UnfoldCommand(mutate))
    except Exception as exc:
        LOG.exception("Unfold failed")
        viewport.flash_status(
            f"Unfold failed: {exc}", 7000
        )
        return False

    notify = getattr(viewport, "notify_scene_changed", None)

    if callable(notify):
        notify()

    viewport.update()

    viewport.flash_status(
        f"Unfold: {len(targets)} face(s) rotated "
        f"{angle_deg:+.1f}°; orphan edges removed.",
        5000,
    )

    return True


class _ClassicMouseFilter(QObject):
    """Implements the double-click / reference-click workflow."""

    def __init__(self, app):
        super().__init__(app.window)

        self.app = app
        self.phase = "idle"
        self.selected_count = 0

    @property
    def viewport(self):
        return self.app.viewport

    def begin(self):
        if self.phase == "idle":
            self.phase = "select"
            self.selected_count = 0

            self.viewport.flash_status(
                "Unfold: DOUBLE-CLICK a face to select "
                "all coplanar faces.",
                5000,
            )

        elif self.phase == "select":
            faces = _selected_faces(self.viewport.scene)

            if not faces:
                self.viewport.flash_status(
                    "Unfold: first double-click a face.",
                    4500,
                )
                return

            self.phase = "reference"
            self.selected_count = len(faces)

            self.viewport.flash_status(
                "Unfold: SINGLE-CLICK the reference face.",
                5000,
            )

        else:
            self.viewport.flash_status(
                "Unfold: SINGLE-CLICK another reference face.",
                4000,
            )

    def _pick_face(self, event):
        p = event.position()

        try:
            return self.viewport.pick_face(
                float(p.x()), float(p.y())
            )
        except Exception:
            LOG.exception("Could not pick face")
            return None

    def _select_coplanar(self, face):
        scene = self.viewport.scene
        faces = _coplanar_faces(scene, face)

        scene.selection.clear()
        scene.selection.update(faces)

        self.selected_count = len(faces)
        self.viewport.update()

        self.viewport.flash_status(
            f"Unfold: {len(faces)} coplanar face(s) selected. "
            "Invoke Unfold again.",
            5000,
        )

    def eventFilter(self, watched, event):
        if watched is not self.viewport:
            return False

        if self.phase == "select":
            if (
                event.type() == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton
            ):
                face = self._pick_face(event)

                if face is not None:
                    self._select_coplanar(face)
                    return True

        elif self.phase == "reference":
            if (
                event.type() == QEvent.Type.MouseButtonPress
                and event.button() == Qt.MouseButton.LeftButton
            ):
                modifiers = event.modifiers()

                if modifiers & (
                    Qt.KeyboardModifier.ShiftModifier
                    | Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                ):
                    return False

                face = self._pick_face(event)

                if face is not None:
                    _unfold_to_reference(self.viewport, face)
                    return True

        return False


def setup(app):
    filt = _ClassicMouseFilter(app)
    app.viewport.installEventFilter(filt)

    menu = app.add_menu("Unfold")

    if menu is not None:
        menu.addAction("Unfold", filt.begin)

    # Keep a reference so the event filter is not garbage-collected.
    app._unfold_classic_filter = filt
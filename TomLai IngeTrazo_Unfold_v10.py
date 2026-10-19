
# SPDX-License-Identifier: GPL-3.0-or-later
"""
IngeTrazo Unfold — Single/Coplanar Selection v10 (clean readable source)

Workflow:
  1. Extensions -> Unfold -> Unfold (first returns the active viewport tool
     to Select so any in-progress Circle/Rectangle/etc. operation is cancelled).
  2. Double-click a face to select its edge-connected coplanar group.
  3. Click the adjacent reference face to unfold.

This revision retains the earlier unfold/rebuild logic. Coplanar selection
uses overlapping boundary-segment adjacency, including differently subdivided
edges on rebuilt faces. Coplanarity uses the strict plane-and-normal test.
Selection changes now increment the scene version as well as notifying/repainting.
Coplanar adjacency also checks native shared-edge face links before geometric
segment overlap, to reduce missed neighbours on rebuilt mesh topology. The workflow
fully detaches on Escape/Space and watches for another active tool being selected.

Controls:
  Escape : Fully exit Unfold and detach its event filter.
  Space  : Fully exit Unfold and detach its event filter.
  Other tool: Automatically exit when a different active tool is detected.

Cancellation does not change geometry.
The unfolding operation remains a single undoable command.

This file contains ordinary readable Python source. It uses no obfuscation,
eval/exec, encoded payloads, or unrelated background tasks.
"""

from __future__ import annotations

import copy
import logging
import math

from PySide6.QtCore import QObject, QEvent, Qt, QTimer
from PySide6.QtGui import QVector3D, QKeySequence, QShortcut
from PySide6.QtWidgets import QApplication

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


def _face_loop_points(face):
    """Yield every vertex position from outer and inner face loops."""
    for loop in (face.loop, *face.hole_loops):
        for vertex in loop:
            yield _q(vertex.position)


def _same_plane(a, b):
    """Strictly test plane alignment and every vertex against both planes.

    Comparing only normals and one point can misclassify a shallow-angled or
    slightly non-planar polygon as coplanar. Checking all loop vertices against
    both planes prevents that shortcut while retaining the original tolerances.
    """
    pa = _face_plane(a)
    pb = _face_plane(b)

    if pa is None or pb is None:
        return False

    na, da = pa
    nb, db = pb

    dot = QVector3D.dotProduct(na, nb)
    if abs(abs(dot) - 1.0) > NORMAL_TOL:
        return False

    # Use absolute plane distances so opposite face-normal directions behave
    # identically. Check both complete faces, not just their first vertices.
    for point in _face_loop_points(a):
        if abs(QVector3D.dotProduct(nb, point) - db) > PLANE_TOL:
            return False
    for point in _face_loop_points(b):
        if abs(QVector3D.dotProduct(na, point) - da) > PLANE_TOL:
            return False

    return True




def _selection_point_key(vertex):
    """Quantised position key used only for selection adjacency/display.

    Rebuilt faces can retain coincident endpoint positions while using
    different vertex objects. Identity-only adjacency then fragments one
    visually continuous coplanar surface into several selectable islands.
    Quantising at 1e-5 model units bridges those seams for selection only;
    unfold/hinge topology continues to use object identity.
    """
    p = _q(vertex.position)
    scale = 100000.0
    return (round(p.x() * scale), round(p.y() * scale), round(p.z() * scale))


def _selection_edge_key(va, vb):
    return frozenset((_selection_point_key(va), _selection_point_key(vb)))


def _edge_segments_overlap(a0, a1, b0, b1, tol=1e-5):
    """Return True when two 3D boundary segments share a non-zero interval.

    Unlike endpoint-key matching, this also recognises a long edge on one face
    adjoining two shorter collinear edges on neighbouring faces.
    """
    a0, a1, b0, b1 = _q(a0), _q(a1), _q(b0), _q(b1)
    av = a1 - a0
    length = av.length()
    if length <= tol:
        return False
    direction = av / length
    bv = b1 - b0

    # Both endpoints of B must lie on A's infinite line, within model tolerance.
    rel0, rel1 = b0 - a0, b1 - a0
    t0 = QVector3D.dotProduct(rel0, direction)
    t1 = QVector3D.dotProduct(rel1, direction)
    off0 = (rel0 - direction * t0).length()
    off1 = (rel1 - direction * t1).length()
    if off0 > tol or off1 > tol:
        return False

    # Reject a degenerate or angled B segment even if its endpoints happen to
    # lie close to the same line at this scale.
    if bv.length() <= tol:
        return False
    cross = QVector3D.crossProduct(direction, bv.normalized()).length()
    if cross > 1e-5:
        return False

    overlap = min(length, max(t0, t1)) - max(0.0, min(t0, t1))
    return overlap > tol


def _face_boundary_segments(face):
    """Yield (start position, end position) for all outer/hole loop edges."""
    for loop in (face.loop, *face.hole_loops):
        if not loop:
            continue
        for i, va in enumerate(loop):
            vb = loop[(i + 1) % len(loop)]
            yield _q(va.position), _q(vb.position)


def _hidden_edge_blocks_segment(mesh, a0, a1, b0, b1, tol=1e-5):
    """Return True if a hidden mesh edge lies along the shared boundary.

    This is used for geometric adjacency between rebuilt faces that do not
    share native vertex/edge objects. Hidden edges remain selection barriers
    even when geometric segment matching is needed to bridge rebuilt seams.
    """
    for edge in mesh.edges:
        if not getattr(edge, "hidden", False):
            continue
        va = getattr(edge, "v0", None)
        vb = getattr(edge, "v1", None)
        if va is None or vb is None:
            continue
        e0, e1 = _q(va.position), _q(vb.position)
        if (_edge_segments_overlap(a0, a1, e0, e1, tol)
                and _edge_segments_overlap(b0, b1, e0, e1, tol)):
            return True
    return False


def _coplanar_faces(scene, ref):
    """Find the edge-connected coplanar region containing ``ref``.

    Face adjacency is based on overlapping boundary segments, not only equal
    endpoint pairs. This handles rebuilt surfaces where one face has a long
    boundary edge and its neighbour has that same boundary split into shorter
    segments. Display flags do not block traversal; coplanarity remains strict.
    """
    mesh = scene.mesh
    if (not isinstance(ref, Face) or ref not in mesh.faces
            or not scene.entity_selectable(ref)):
        return []

    faces = [f for f in mesh.faces
             if isinstance(f, Face) and scene.entity_selectable(f)]
    segments = {f: list(_face_boundary_segments(f)) for f in faces}
    adjacency = {f: set() for f in faces}
    face_set = set(faces)

    # First use the mesh's own edge-to-face topology. This is the most reliable
    # source when the rebuilt faces really share native edge objects.
    for edge in mesh.edges:
        # Hidden edges are explicit barriers: do not flood through them even
        # when the faces on either side happen to be coplanar.
        if getattr(edge, "hidden", False):
            continue
        linked = [f for f in getattr(edge, "faces", ()) if f in face_set]
        if len(linked) < 2:
            continue
        for i, face_a in enumerate(linked):
            for face_b in linked[i + 1:]:
                if _same_plane(face_a, face_b):
                    adjacency[face_a].add(face_b)
                    adjacency[face_b].add(face_a)

    # Also compare geometric boundary segments. This bridges seams where
    # rebuilt faces are coincident but do not share native vertex/edge objects,
    # and where a long edge adjoins multiple shorter collinear edges.
    for i, face_a in enumerate(faces):
        segs_a = segments[face_a]
        if not segs_a:
            continue
        for face_b in faces[i + 1:]:
            if not _same_plane(face_a, face_b):
                continue
            segs_b = segments[face_b]
            connected = False
            for a0, a1 in segs_a:
                for b0, b1 in segs_b:
                    if not _edge_segments_overlap(a0, a1, b0, b1):
                        continue
                    if _hidden_edge_blocks_segment(mesh, a0, a1, b0, b1):
                        continue
                    connected = True
                    break
                if connected:
                    break
            if connected:
                adjacency[face_a].add(face_b)
                adjacency[face_b].add(face_a)

    selected = {ref}
    pending = [ref]
    while pending:
        current = pending.pop()
        for neighbour in adjacency.get(current, ()):
            if neighbour not in selected:
                selected.add(neighbour)
                pending.append(neighbour)

    result = [face for face in mesh.faces if face in selected]
    LOG.info("Coplanar selection: %d face(s) using segment-overlap adjacency.", len(result))
    return result


def _make_selection_boundary_hard_visible(scene, selected_faces):
    """Make the selected group's perimeter edges hard and explicitly visible.

    Perimeter means an edge used by a selected face that is also used by an
    unselected face, or that has no other face (an outer boundary). Hidden
    edges used only internally by the selected group are left unchanged.
    """
    mesh = scene.mesh
    selected = set(selected_faces)
    if not selected:
        return 0

    mesh_edges = {}
    for edge in mesh.edges:
        va = getattr(edge, "v0", None)
        vb = getattr(edge, "v1", None)
        if va is not None and vb is not None:
            mesh_edges[_selection_edge_key(va, vb)] = edge

    edge_faces = {}
    for face in mesh.faces:
        if not isinstance(face, Face):
            continue
        for loop in (face.loop, *face.hole_loops):
            if not loop:
                continue
            for i, va in enumerate(loop):
                vb = loop[(i + 1) % len(loop)]
                edge_faces.setdefault(_selection_edge_key(va, vb), set()).add(face)

    changed_edges = set()
    boundary_edge_objects = []
    for face in selected:
        for loop in (face.loop, *face.hole_loops):
            if not loop:
                continue
            for i, va in enumerate(loop):
                vb = loop[(i + 1) % len(loop)]
                key = _selection_edge_key(va, vb)
                adjacent = edge_faces.get(key, set())
                if adjacent and adjacent.issubset(selected) and len(adjacent) > 1:
                    continue
                edge = mesh_edges.get(key)
                if edge is None or id(edge) in changed_edges:
                    continue
                changed_edges.add(id(edge))
                boundary_edge_objects.append(edge)
                try:
                    edge.hidden = False
                except Exception:
                    LOG.debug("Could not unhide selected-group boundary edge", exc_info=True)
                try:
                    edge.soft = False
                except Exception:
                    LOG.debug("Could not harden selected-group boundary edge", exc_info=True)

    LOG.info("Made %d selected-group boundary edge(s) hard and visible.", len(changed_edges))
    return boundary_edge_objects


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








def _copy_face_data(face):
    return copy.deepcopy(face.attrs or {})


def _selected_faces(scene):
    return [
        item for item in scene.selection
        if isinstance(item, Face)
        and item in scene.mesh.faces
    ]


def _delete_orphan_edges(mesh):
    """Delete edges that no longer belong to any face."""
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

    # Find the actual shared edge between reference and selection.
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
            min(1.0, QVector3D.dotProduct(n_tar, n_ref)),
        ),
    ))

    # Check IngeTrazo's rotation-matrix sign convention.
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

    # Preserve original face geometry and attributes.
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

        for face in targets:
            if face in current_mesh.faces:
                current_mesh.remove_face(face)

        orphan_count = _delete_orphan_edges(current_mesh)
        rebuilt = []

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

        # Make all edges belonging to the unfolded sheet hard and visible.
        # IngeTrazo may otherwise retain hidden/soft flags on rebuilt edges,
        # particularly around high-sided polygonal faces.
        hard_visible_count = 0
        processed_edges = set()

        def make_hard_visible(edge):
            nonlocal hard_visible_count
            if edge is None or id(edge) in processed_edges:
                return
            processed_edges.add(id(edge))
            changed = False
            try:
                if getattr(edge, "soft", False):
                    edge.soft = False
                    changed = True
            except Exception:
                LOG.debug("Could not clear soft flag on edge", exc_info=True)
            try:
                if getattr(edge, "hidden", False):
                    edge.hidden = False
                    changed = True
            except Exception:
                LOG.debug("Could not clear hidden flag on edge", exc_info=True)
            # Count edges touched, not only edges whose flags changed.
            hard_visible_count += 1

        def find_loop_edge(va, vb):
            edge = current_mesh.find_edge(va, vb)
            if edge is not None:
                return edge
            # Fallback to direct endpoint identity if find_edge misses it.
            for candidate in current_mesh.edges:
                e0 = getattr(candidate, "v0", None)
                e1 = getattr(candidate, "v1", None)
                if (e0 is va and e1 is vb) or (e0 is vb and e1 is va):
                    return candidate
            return None

        # Every edge of a rebuilt face includes the former hinge junctions.
        for face in rebuilt:
            for loop in (face.loop, *face.hole_loops):
                for i, va in enumerate(loop):
                    vb = loop[(i + 1) % len(loop)]
                    make_hard_visible(find_loop_edge(va, vb))

        # Also harden/restore visibility on the reference face boundary wherever
        # it meets a genuinely non-coplanar neighbouring face.  This makes the
        # boundary legible even when the neighbour has hidden-edge flags.
        reference_boundary_count = 0
        if ref in current_mesh.faces:
            for loop in (ref.loop, *ref.hole_loops):
                for i, va in enumerate(loop):
                    vb = loop[(i + 1) % len(loop)]
                    edge = find_loop_edge(va, vb)
                    if edge is None:
                        continue
                    neighbours = [
                        other for other in getattr(edge, "faces", ())
                        if other is not ref and other in current_mesh.faces
                    ]
                    if any(not _same_plane(ref, other) for other in neighbours):
                        make_hard_visible(edge)
                        reference_boundary_count += 1

        current_scene.selection.clear()
        selected_after_unfold = list(rebuilt)
        if ref in current_mesh.faces:
            selected_after_unfold.append(ref)
        current_scene.selection.update(selected_after_unfold)

        LOG.info(
            "Unfold rebuilt %d faces; selected %d faces after unfold; "
            "made %d unfolded-sheet edges hard/visible; processed %d "
            "reference boundary edge(s) beside non-coplanar faces; "
            "removed %d orphan edges.",
            len(rebuilt),
            len(selected_after_unfold),
            hard_visible_count,
            reference_boundary_count,
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


def _mark_selection_changed(scene):
    """Invalidate scene/selection render caches after a selection-only change.

    Geometry operations pass through history and increment scene.version. The
    interactive selection path does not, which can leave the renderer showing
    the previous selection until the command exits or geometry changes.
    """
    try:
        scene.version += 1
    except Exception:
        LOG.debug("Could not increment scene version for selection", exc_info=True)


def _refresh_selection_viewport(viewport):
    """Ask the viewport to redraw after selection and edge-visibility changes."""
    notify = getattr(viewport, "notify_scene_changed", None)
    if callable(notify):
        try:
            notify()
        except Exception:
            LOG.debug("Selection scene notification failed", exc_info=True)

    # update() schedules a paint; repaint() additionally requests an immediate
    # paint where supported. Queue one more update after the mouse event has
    # returned so the viewport's native selection overlay can rebuild.
    try:
        viewport.update()
    except Exception:
        LOG.debug("Viewport update after selection failed", exc_info=True)
    repaint = getattr(viewport, "repaint", None)
    if callable(repaint):
        try:
            repaint()
        except Exception:
            LOG.debug("Immediate viewport repaint failed", exc_info=True)
    QTimer.singleShot(0, viewport.update)
    # Some viewport builds rebuild selection/render caches only after the mouse
    # event has unwound. Notify and repaint once more after that event boundary.
    def _late_refresh():
        late_notify = getattr(viewport, "notify_scene_changed", None)
        if callable(late_notify):
            try:
                late_notify()
            except Exception:
                LOG.debug("Deferred scene notification failed", exc_info=True)
        try:
            viewport.update()
        except Exception:
            LOG.debug("Deferred viewport update failed", exc_info=True)
    QTimer.singleShot(40, _late_refresh)


class _ClassicMouseFilter(QObject):
    """Single-click one face or double-click a coplanar group, then unfold."""

    def __init__(self, app):
        super().__init__(app.window)

        self.app = app
        self.phase = "idle"
        self.selected_count = 0
        self._pending_single_face = None
        self._ignore_release_after_double = False
        self._filter_installed = False
        self._tool_state_at_begin = None
        self._single_click_timer = QTimer(app.window)
        self._single_click_timer.setSingleShot(True)
        self._single_click_timer.timeout.connect(self._finish_single_click)
        try:
            interval = QApplication.styleHints().mouseDoubleClickInterval()
        except Exception:
            interval = 250
        # Qt emits MouseButtonDblClick only after the first release and the
        # second press. Do not let the single-click timer race Qt's own
        # double-click recognizer at the exact same deadline.
        self._double_click_interval = max(150, int(interval))
        self._single_click_timer.setInterval(self._double_click_interval + 100)

        # Poll the viewport's active-tool state while this workflow is active.
        # The classic workflow is an event filter rather than a native Tool, so
        # it must detach itself when the user activates a different tool.
        self._tool_watch_timer = QTimer(app.window)
        self._tool_watch_timer.setInterval(150)
        self._tool_watch_timer.timeout.connect(self._check_active_tool)

        # Shortcuts are children of the main window and work when
        # keyboard focus is within that window's child widgets.
        self.escape_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Escape),
            app.window,
        )
        self.space_shortcut = QShortcut(
            QKeySequence(Qt.Key.Key_Space),
            app.window,
        )

        self.escape_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )
        self.space_shortcut.setContext(
            Qt.ShortcutContext.WidgetWithChildrenShortcut
        )

        self.escape_shortcut.activated.connect(
            self._cancel_workflow
        )
        self.space_shortcut.activated.connect(
            self._cancel_workflow
        )

        self._update_shortcuts()

    @property
    def viewport(self):
        return self.app.viewport

    def _update_shortcuts(self):
        active = self.phase != "idle"
        self.escape_shortcut.setEnabled(active)
        self.space_shortcut.setEnabled(active)

    def _install_event_filter(self):
        if not self._filter_installed:
            self.viewport.installEventFilter(self)
            self._filter_installed = True

    def _remove_event_filter(self):
        if self._filter_installed:
            try:
                self.viewport.removeEventFilter(self)
            except Exception:
                LOG.debug("Could not detach Unfold event filter", exc_info=True)
            self._filter_installed = False

    def _active_tool_state(self):
        """Return a comparable signature for the viewport's active tool.

        IngeTrazo builds may expose the current tool under different attribute
        names. Probe common public/private forms without assuming one exists.
        Return None if this build exposes no readable active-tool state.
        """
        owners = [self.viewport, self.app]
        for owner_name in ("tool_manager", "tools", "tool_controller"):
            for owner in (self.viewport, self.app):
                try:
                    child = getattr(owner, owner_name, None)
                except Exception:
                    child = None
                if child is not None and not callable(child):
                    owners.append(child)

        names = (
            "active_tool", "_active_tool", "current_tool", "_current_tool",
            "tool", "_tool", "activeTool", "currentTool",
        )
        found = []
        seen_owners = set()
        for owner in owners:
            if owner is None or id(owner) in seen_owners:
                continue
            seen_owners.add(id(owner))
            for name in names:
                try:
                    value = getattr(owner, name, None)
                except Exception:
                    continue
                if value is None or callable(value):
                    continue
                # Include the attribute name and object identity. Tool objects
                # are compared by identity, not by potentially mutable equality.
                found.append((id(owner), name, id(value), type(value).__name__))
        return tuple(found) if found else None

    def _check_active_tool(self):
        if self.phase == "idle":
            self._tool_watch_timer.stop()
            return
        current = self._active_tool_state()
        changed = (
            current is not None and current != self._tool_state_at_begin
        ) or (
            self._tool_state_at_begin is not None and current is None
        )
        if changed:
            self._cancel_workflow(message="Unfold exited because another tool was selected.")

    def _handoff_active_tool_to_select(self):
        """Deactivate any currently active modelling tool before Unfold starts.

        Unfold is implemented as a viewport event filter rather than a native
        IngeTrazo Tool. If Circle, Rectangle, Line, or another native tool stays
        active underneath this filter, both tools can respond to the same input.
        The application's normal tool router activates Select, which should
        trigger the previous tool's normal deactivation/cancellation lifecycle.
        """
        window = getattr(self.app, "window", None)
        if callable(window):
            try:
                window = window()
            except Exception:
                window = None
        if window is None:
            window = self.app

        activate_tool = getattr(window, "_activate_tool", None)
        if callable(activate_tool):
            try:
                activate_tool("select")
                return True
            except Exception:
                LOG.debug(
                    "Could not switch the active tool to Select before Unfold",
                    exc_info=True,
                )

        # Some builds may expose the tool router on the viewport instead.
        activate_tool = getattr(self.viewport, "_activate_tool", None)
        if callable(activate_tool):
            try:
                activate_tool("select")
                return True
            except Exception:
                LOG.debug(
                    "Viewport tool-router fallback failed before Unfold",
                    exc_info=True,
                )

        return False

    def begin(self):
        if self.phase == "idle":
            # First deactivate any native drawing/modelling tool. This must
            # happen before taking the active-tool snapshot, otherwise the
            # watchdog could mistake our own handoff to Select for a later
            # external tool switch and immediately cancel Unfold.
            self._handoff_active_tool_to_select()

            self._install_event_filter()
            self.phase = "select"
            self.selected_count = 0
            self._tool_state_at_begin = self._active_tool_state()
            # Keep watching even if this build exposes no active tool initially;
            # a later readable tool state still signals a tool switch.
            self._tool_watch_timer.start()
            self._update_shortcuts()

            self.viewport.flash_status(
                "Unfold: single-click one face or double-click a coplanar group; "
                "then click the adjacent reference face. Repeat without restarting. "
                "Esc/Space cancels. Any previous tool has been returned to Select.",
                6000,
            )

        elif self.phase == "select":
            self.viewport.flash_status(
                "Unfold: single-click one face or double-click a coplanar group; "
                "then click its adjacent reference face. Esc/Space cancels.",
                4500,
            )

        else:
            self.viewport.flash_status(
                "Unfold: click a reference face, "
                "or press Esc/Space to cancel.",
                4000,
            )

    def _cancel_workflow(self, message="Unfold cancelled."):
        """Fully detach the active workflow without modifying geometry."""
        if self.phase == "idle":
            return

        self.phase = "idle"
        self._single_click_timer.stop()
        self._tool_watch_timer.stop()
        self._pending_single_face = None
        self._ignore_release_after_double = False
        self.selected_count = 0
        self._tool_state_at_begin = None
        self._update_shortcuts()
        self._remove_event_filter()

        # Clear native hover preview left by reference-face mode, if supported.
        set_hover = getattr(self.viewport, "set_hover", None)
        if callable(set_hover):
            try:
                set_hover(None)
            except Exception:
                LOG.debug("Could not clear hover state on exit", exc_info=True)
        _refresh_selection_viewport(self.viewport)
        self.viewport.flash_status(message, 3000)

    def _pick_face(self, event):
        p = event.position()

        try:
            return self.viewport.pick_face(
                float(p.x()), float(p.y())
            )
        except Exception:
            LOG.exception("Could not pick face")
            return None

    def _finish_single_click(self):
        """Resolve a click only after the double-click window has elapsed."""
        face = self._pending_single_face
        self._pending_single_face = None
        if self.phase != "select" or face is None:
            return
        self._select_single(face)

    def _select_single(self, face):
        """Select exactly the clicked face, without expanding to neighbours."""
        scene = self.viewport.scene
        if (not isinstance(face, Face) or face not in scene.mesh.faces
                or not scene.entity_selectable(face)):
            return

        # Select the face plus its perimeter edges. In some viewport builds
        # the face-selection dotted overlay is absent or only partly rendered;
        # the selected edge outline makes a single-face selection unambiguous.
        boundary_edges = _make_selection_boundary_hard_visible(scene, [face])
        scene.selection.clear()
        scene.selection.add(face)
        scene.selection.update(boundary_edges)
        _mark_selection_changed(scene)
        self.selected_count = 1
        self.phase = "reference"
        self._update_shortcuts()
        _refresh_selection_viewport(self.viewport)
        self.viewport.flash_status(
            "Unfold: 1 face selected. Click an adjacent reference face to unfold; "
            "Esc/Space cancels.",
            5000,
        )

    def _select_coplanar(self, face):
        scene = self.viewport.scene
        faces = _coplanar_faces(scene, face)
        if not faces:
            self.viewport.flash_status(
                "Unfold: no selectable coplanar faces found. Click a face and try again.",
                4500,
            )
            return

        # Make the selected group's perimeter explicit in the viewport. This
        # is intentionally limited to boundary edges; hidden internal edges do
        # not become selection links and are not globally exposed.
        boundary_edges = _make_selection_boundary_hard_visible(scene, faces)

        scene.selection.clear()
        scene.selection.update(faces)
        # Explicitly select the perimeter edges as a crisp outline. In some
        # IngeTrazo builds only one face in a multi-face selection receives the
        # dotted face overlay; the edge outline makes the whole selected region
        # legible without creating temporary duplicate/thickened geometry.
        scene.selection.update(boundary_edges)
        _mark_selection_changed(scene)

        self.selected_count = len(faces)

        # Enter reference mode before requesting repaint, then force both the
        # native selection overlay and explicit edge outline to redraw.
        # Keep the actual selection as faces plus boundary edges.
        # Enter reference mode for this group. After unfolding, the workflow
        # returns to selection mode so another group can be chosen without
        # invoking Unfold again.
        self.phase = "reference"
        self._update_shortcuts()
        _refresh_selection_viewport(self.viewport)

        self.viewport.flash_status(
            f"Unfold: {len(faces)} coplanar face(s) selected; "
            f"{len(boundary_edges)} boundary edge(s) outlined and made hard/visible. "
            "Hidden edges act as selection barriers. Move "
            "over the adjacent reference face to preview it, then click. "
            "Esc/Space cancels.",
            6000,
        )

    def eventFilter(self, watched, event):
        if watched is not self.viewport:
            return False

        # Fallback key handling if the viewport receives key events.
        if event.type() == QEvent.Type.KeyPress:
            if event.key() in (
                Qt.Key.Key_Escape,
                Qt.Key.Key_Space,
            ):
                if self.phase != "idle":
                    self._cancel_workflow()
                    return True

                return False

        # Do not let native viewport drag-selection receive left-button move
        # events while this event-filter tool is active. A reference-face press
        # being passed through to the viewport can otherwise leave a screen-space
        # blue/green marquee anchored at the click position after the unfold.
        if (self.phase != "idle"
                and event.type() == QEvent.Type.MouseMove
                and event.buttons() & Qt.MouseButton.LeftButton):
            return True

        # The second release in Qt's double-click sequence arrives after the
        # MouseButtonDblClick event. Consume it so it cannot be mistaken for a
        # click on the reference face after the phase changes.
        if (self._ignore_release_after_double
                and event.type() == QEvent.Type.MouseButtonRelease
                and event.button() == Qt.MouseButton.LeftButton):
            self._ignore_release_after_double = False
            return True

        if self.phase == "select":
            if (event.type() == QEvent.Type.MouseButtonPress
                    and event.button() == Qt.MouseButton.LeftButton):
                modifiers = event.modifiers()
                if modifiers & (
                    Qt.KeyboardModifier.ShiftModifier
                    | Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                ):
                    return False
                # Keep native viewport selection from competing with this tool.
                return True

            if (event.type() == QEvent.Type.MouseButtonRelease
                    and event.button() == Qt.MouseButton.LeftButton):
                modifiers = event.modifiers()
                if modifiers & (
                    Qt.KeyboardModifier.ShiftModifier
                    | Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                ):
                    return False
                self._pending_single_face = self._pick_face(event)
                if self._pending_single_face is not None:
                    self._single_click_timer.start()
                return True

            if (
                event.type() == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton
            ):
                modifiers = event.modifiers()
                if modifiers & (
                    Qt.KeyboardModifier.ShiftModifier
                    | Qt.KeyboardModifier.ControlModifier
                    | Qt.KeyboardModifier.AltModifier
                ):
                    return False
                self._single_click_timer.stop()
                self._pending_single_face = None
                self._ignore_release_after_double = True
                face = self._pick_face(event)
                if face is not None:
                    self._select_coplanar(face)
                return True

        elif self.phase == "reference":
            if (event.type() == QEvent.Type.MouseButtonRelease
                    and event.button() == Qt.MouseButton.LeftButton):
                return True

            # A double-click in reference mode must never fall through to
            # IngeTrazo's native double-click selection. Native selection can
            # expand across hidden edges into a connected but non-coplanar
            # block, which would contaminate the selected faces and allow an
            # unintended block unfold on a subsequent click. The first press
            # already performs the normal single-click reference-face check;
            # consume the double-click event and its trailing release here.
            if (
                event.type() == QEvent.Type.MouseButtonDblClick
                and event.button() == Qt.MouseButton.LeftButton
            ):
                self._single_click_timer.stop()
                self._pending_single_face = None
                self._ignore_release_after_double = True
                return True

            # Use IngeTrazo's native hover/preselection highlight for the face
            # currently under the pointer. This is a display-only cue; it does
            # not alter the selected coplanar group or model geometry.
            if event.type() == QEvent.Type.MouseMove:
                face = self._pick_face(event)
                set_hover = getattr(self.viewport, "set_hover", None)
                if callable(set_hover):
                    try:
                        set_hover(face)
                        self.viewport.update()
                    except Exception:
                        LOG.debug("Reference-face hover update failed", exc_info=True)
                return False

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
                    selected_faces = _selected_faces(self.viewport.scene)
                    if face in selected_faces:
                        self.viewport.flash_status(
                            "Unfold: click an adjacent, unselected reference face.",
                            3500,
                        )
                        return True

                    success = _unfold_to_reference(self.viewport, face)

                    if success:
                        # Keep the command active, but begin a fresh selection
                        # cycle for the next coplanar group.
                        self.phase = "select"
                        self.selected_count = 0
                        self._update_shortcuts()
                        self.viewport.flash_status(
                            "Unfold complete. Single-click the next face or double-click "
                            "the next coplanar group, then click its adjacent reference face. "
                            "Esc/Space exits.",
                            6000,
                        )
                    # If unsuccessful, do not replace the detailed reason shown
                    # by _unfold_to_reference; remain in reference mode for retry.
                    # Always consume the press so the native viewport cannot
                    # start a drag-selection rectangle from this click.
                    return True

                else:
                    # Clicking empty space while waiting for a reference face
                    # abandons the current selection and starts a fresh selection
                    # cycle. Without this reset, a stale reference phase can
                    # remain active with no valid selected faces, causing later
                    # clicks to repeatedly report "no selected faces to unfold".
                    scene = self.viewport.scene
                    scene.selection.clear()
                    _mark_selection_changed(scene)
                    self.selected_count = 0
                    self.phase = "select"
                    self._single_click_timer.stop()
                    self._pending_single_face = None
                    set_hover = getattr(self.viewport, "set_hover", None)
                    if callable(set_hover):
                        try:
                            set_hover(None)
                        except Exception:
                            LOG.debug(
                                "Could not clear hover after empty-space click",
                                exc_info=True,
                            )
                    self._update_shortcuts()
                    _refresh_selection_viewport(self.viewport)
                    self.viewport.flash_status(
                        "Unfold reset. Single-click a face or double-click a "
                        "coplanar group to begin again.",
                        4500,
                    )

                    return True

        # While active, leave non-click events to the viewport. In particular,
        # do not change phase or cancel on mouse move/release events.
        return False


def setup(app):
    filt = _ClassicMouseFilter(app)

    menu = app.add_menu("Unfold")

    if menu is not None:
        menu.addAction("Unfold", filt.begin)

    # Keep a reference so the filter and shortcuts remain alive.
    app._unfold_classic_filter = filt

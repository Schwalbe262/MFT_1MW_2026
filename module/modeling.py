from pyaedt_module.model3d.core import Core
from pyaedt_module.model3d.transformer_winding import Transformer_winding
import math



def create_core(design, name="core", core_material="ferrite") :

    main = design.modeler.create_box(
        origin = ["-(4*l1+2*l2)/2", "-w1/2", "-(h1+2*l1)/2"],
        sizes = ["4*l1+2*l2", "w1", "h1+2*l1"],
        name = f"{name}",
        material = core_material
    )

    sub1 = design.modeler.create_box(
        origin = ["-l1", "-w1/2", "-h1/2"],
        sizes = ["-l2", "w1", "h1"],
        name = f"{name}_sub1",
        material = core_material
    )

    sub2 = design.modeler.create_box(
        origin = ["l1", "-w1/2", "-h1/2"],
        sizes = ["l2", "w1", "h1"],          
        name = f"{name}_sub2",
        material = core_material
    )

    
    design.modeler.subtract(
        [main],
        [sub1, sub2],
        keep_originals=False
    )

    return main


def create_coil(design, name, window_height, window_length, window_layer, N_input, width_fill_factor, space_length, space_width, shape="circle", offset = [0,0,0], color = None):

    # name : coil의 name
    # window_height : 권선 윈도우의 높이
    # 
    coil_width = 0
    coil_height = 0
    coil_gap_x = 0
    coil_gap_z = 0

    if N_input != None :
        N = N_input

    shape = shape.lower()

    # 1차 측 코일일
    if shape == "circle" :

        if window_layer > 1 :
            coil_width = window_length * width_fill_factor / window_layer
            coil_height = coil_width
        elif window_layer == 1 :
            coil_width = window_length
            coil_height = coil_width
    

        if window_layer > 1 :
            coil_gap_x = (window_length - (coil_width * window_layer)) / (window_layer - 1)
        else :
            coil_gap_x = 0

        # N = int(window_height / coil_width * height_fill_factor)
        coil_gap_z = (window_height - (coil_width * N)) / (N - 1)



    elif shape == "rectangle" :

        coil_width = window_length * width_fill_factor / window_layer
        coil_height = window_height / N

        if window_layer > 1 :
            coil_gap_x = (window_length - (coil_width * window_layer)) / (window_layer - 1)
        else :
            coil_gap_x = 0

        if N > 1 :
            coil_gap_z = (window_height - (coil_height * N)) / (N - 1)
        else :
            coil_gap_z = 0


    x_pos = []
    y_pos = []
    z_pos = []

    for i in range(window_layer) :
        x_pos.append(space_length/2 + coil_width*(i+0.5) + coil_gap_x*i)
        y_pos.append(space_width/2 + coil_width*(i+0.5) + coil_gap_x*i)

    for i in range(N) :
        z_pos.append(window_height/2 - coil_height*(i+0.5) - coil_gap_z*i)

    windings = []

    i = 0
    j = 0

    for i, (x, y) in enumerate(zip(x_pos, y_pos)):
        for j, z in enumerate(z_pos):
                
            x_pos_s = x
            x_neg_s = -x
            y_pos_s = y
            y_neg_s = -y

            points = []
            points.append([f"{x_pos_s}mm + {offset[0]}mm", f"{y_pos_s}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm" ])
            points.append([f"{x_neg_s}mm + {offset[0]}mm", f"{y_pos_s}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm"])
            points.append([f"{x_neg_s}mm + {offset[0]}mm", f"{y_neg_s}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm"])
            points.append([f"{x_pos_s}mm + {offset[0]}mm", f"{y_neg_s}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm"])
            points.append([f"{x_pos_s}mm + {offset[0]}mm", f"{y_pos_s}mm + {offset[1]}mm", f"{z}mm + {offset[2]}mm" ])


            winding = design.modeler.create_polyline(
                points=points, name=f"{name}_{i}_{j}", material="copper",xsection_orient="Auto",
                xsection_type=shape, xsection_width=coil_width, xsection_height=coil_height, xsection_num_seg=6, xsection_topwidth=coil_width)
            if color != None :
                winding.color = color
            windings.append(winding)

    print(f"N = {N}")
    print(f"coil_width = {coil_width}")
    print(f"coil_height = {coil_height}")
    print(f"coil_gap_x = {coil_gap_x}")
    print(f"coil_gap_z = {coil_gap_z}")

    return windings, N, coil_width, coil_height,coil_gap_x, coil_gap_z


def create_coil_section(design, winding_obj, sheet_prefix = None, plane = "ZX", rename_faces = False):


    modeler = design.modeler

    def _to_obj_list(obj_or_list):
        if isinstance(obj_or_list, (list, tuple, set)):
            return list(obj_or_list)
        return [obj_or_list]


    def _center_x(obj):
        bb = getattr(obj, "bounding_box", None)
        if bb and len(bb) == 6:
            return (float(bb[0]) + float(bb[3])) / 2.0
        return 0.0

    def _rename_object_safe(obj, target_name, used_names):
        current_name = getattr(obj, "name", None)
        if not current_name:
            return None

        # If the requested name is already used by another object, append an index.
        candidate = target_name
        if candidate != current_name:
            idx = 1
            while candidate in used_names:
                candidate = f"{target_name}_{idx}"
                idx += 1

        obj.name = candidate
        used_names.discard(current_name)
        used_names.add(candidate)
        return candidate


    # input normalize
    winding_objs = _to_obj_list(winding_obj)
    if not winding_objs:
        raise ValueError("winding_obj is empty")

    if sheet_prefix is None:
        first_name = getattr(winding_objs[0], "name", "winding")
        sheet_prefix = f"{first_name}_sec"

    # 1) section
    before_sheet_names = set(modeler.sheet_names)
    ok = modeler.section(winding_objs, plane)  # plane choices: "XY","YZ","ZX"
    if not ok:
        raise RuntimeError("section failed")

    after_sheet_names = set(modeler.sheet_names)
    new_sheet_names = sorted(after_sheet_names - before_sheet_names)
    sheets = [modeler.get_object_from_name(n) for n in new_sheet_names]

    # 2) separate bodies (collect robustly to avoid missing Separate1)
    before_sep_sheet_names = set(modeler.sheet_names)
    separated = modeler.separate_bodies(assignment=sheets)
    after_sep_sheet_names = set(modeler.sheet_names)
    new_sep_sheet_names = sorted(after_sep_sheet_names - before_sep_sheet_names)

    face_candidates = []
    if isinstance(separated, list):
        face_candidates.extend(separated)

    for n in new_sep_sheet_names:
        o = modeler.get_object_from_name(n)
        if o is not None:
            face_candidates.append(o)

    # fallback include original section sheets too
    face_candidates.extend(sheets)

    # deduplicate by name
    faces = []
    seen = set()
    for o in face_candidates:
        if o is None:
            continue
        n = getattr(o, "name", None)
        if not n or n in seen:
            continue
        seen.add(n)
        faces.append(o)

    # 3) split by x: smaller-x list and larger-x list
    x_small_faces = []
    x_large_faces = []
    used = set()

    for src in winding_objs:
        src_name = str(getattr(src, "name", src)).lower()
        group = [f for f in faces if src_name in str(getattr(f, "name", "")).lower()]
        if len(group) >= 2:
            group_sorted = sorted(group, key=_center_x)
            x_small_faces.append(group_sorted[0])
            x_large_faces.append(group_sorted[-1])
            used.add(group_sorted[0].name)
            used.add(group_sorted[-1].name)

    # if some faces are left unmatched, pair by x order
    remaining = [f for f in faces if getattr(f, "name", "") not in used]
    remaining_sorted = sorted(remaining, key=_center_x)
    for i in range(0, len(remaining_sorted) - 1, 2):
        x_small_faces.append(remaining_sorted[i])
        x_large_faces.append(remaining_sorted[i + 1])

    # 4) optional rename for separated faces
    if rename_faces:
        used_names = set(modeler.object_names)
        pair_count = min(len(x_small_faces), len(x_large_faces))
        for i in range(pair_count):
            if pair_count == 1:
                base = sheet_prefix
            else:
                base = f"{sheet_prefix}_{i + 1}"

            _rename_object_safe(x_small_faces[i], f"{base}_neg", used_names)
            _rename_object_safe(x_large_faces[i], f"{base}_pos", used_names)

    # x_neg와 x_pos는 각각 2개의 separated face 중 x 값이 작은/큰 face의 name list
    x_neg = [f.name for f in x_small_faces]
    x_pos = [f.name for f in x_large_faces]

    return x_neg, x_pos
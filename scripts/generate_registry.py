import sys
import os
import clang.cindex
import errno
import shlex
import re

def generate():
    output_path = sys.argv[1]
    base_class = sys.argv[2]
    headers = sys.argv[3:]
    
    base_classes = []
    
    if "," in base_class:
        base_classes = base_class.split(",")
    else:
        base_classes = [base_class]

    # Libclang initialisieren (HPC Pfad)
    # clang.cindex.Config.set_library_path(os.environ.get('EBROOTCLANG') + '/lib')

    # Some systems / Python environments can't find libclang automatically.
    # Try to create the Index and if that fails, try a short list of
    # common libclang locations and a user-provided LIBCLANG_PATH env var.
    try:
        index = clang.cindex.Index.create()
    except clang.cindex.LibclangError:
        candidates = [
            os.environ.get('LIBCLANG_PATH'),
            '/usr/lib/llvm-18/lib/libclang.so.1',
            '/usr/lib/x86_64-linux-gnu/libclang-18.so.18',
            '/usr/lib/x86_64-linux-gnu/libclang-18.so.1',
            '/usr/lib/llvm-15/lib/libclang.so.1',
            '/usr/lib/llvm-11/lib/libclang.so.1',
        ]

        set_success = False
        for p in candidates:
            if not p:
                continue
            try:
                if os.path.exists(p):
                    clang.cindex.Config.set_library_file(p)
                    index = clang.cindex.Index.create()
                    set_success = True
                    break
            except OSError as e:
                # If the path exists but loading fails, continue to next
                if getattr(e, 'errno', None) in (errno.ENOENT,):
                    continue
        if not set_success:
            # re-raise original error to preserve original traceback
            raise
    found_classes = dict()
    base_classes_found = dict()
    subclass_constructors = dict()  # Store constructor info for each subclass
    template_parameters = dict()  # Store template parameters: {class_name: [param_names]}
    
    for base_class in base_classes:
        found_classes[base_class] = []
        subclass_constructors[base_class] = dict()  # {subclass_name: [constructors]}
        template_parameters[base_class] = []

    # Allow user to supply additional clang args via CLANG_ARGS env var
    env_clang_args = os.environ.get('CLANG_ARGS')
    if env_clang_args:
        try:
            parse_args = shlex.split(env_clang_args)
        except Exception:
            parse_args = ['-std=c++17']
    else:
        # default: use c++17 and the project include dir
        parse_args = ['-std=c++17', '-I.', '-Iinclude']

    def get_template_parameters(node):
        """Extract template parameter names from a class template node"""
        template_params = []
        if node.kind == clang.cindex.CursorKind.CLASS_TEMPLATE:  # type: ignore
            for child in node.get_children():
                if child.kind == clang.cindex.CursorKind.TEMPLATE_TYPE_PARAMETER:  # type: ignore
                    param_name = child.spelling or f"T{len(template_params)}"
                    template_params.append(param_name)
                elif child.kind == clang.cindex.CursorKind.TEMPLATE_NON_TYPE_PARAMETER:  # type: ignore
                    param_name = child.spelling or f"N{len(template_params)}"
                    template_params.append(param_name)
        return template_params

    def is_containing_class_declaration(node, class_name, current_file):
        # Return tuple: (is_containing, constructors)
        # is_containing: True if this node declares the class named `class_name` in current file
        # constructors: list of constructor parameter lists if found
        if node.kind in (clang.cindex.CursorKind.CLASS_DECL, # type: ignore
                         clang.cindex.CursorKind.STRUCT_DECL, # type: ignore
                         clang.cindex.CursorKind.CLASS_TEMPLATE): # type: ignore
            # Get the node's actual name (without template args)
            name = node.spelling or ""
            name = re.sub(r'<.*>$', '', name).strip()
            simple = name.split('::')[-1]
            
            # Check if this node matches the class name AND is defined in the current file
            if simple == class_name or name == class_name:
                node_file = str(node.location.file) if node.location.file else ""
                if node_file.endswith(current_file) or current_file.endswith(node_file):
                    print("  "+ node.spelling + f" (defined in {node_file})")
                    # Extract template parameters for this base class
                    template_params = get_template_parameters(node)
                    if template_params:
                        print(f"    Template parameters: {template_params}")
                    return True, template_params
        return False, []

    def get_class_constructors(node, class_name):
        # Extract constructor information from a class node
        constructors = []
        for child in node.get_children():
            if child.kind == clang.cindex.CursorKind.CONSTRUCTOR:  # type: ignore
                # Get parameter list for this constructor
                params = []
                for param in child.get_children():
                    if param.kind == clang.cindex.CursorKind.PARM_DECL:  # type: ignore
                        param_type = param.type.spelling if param.type else "unknown"
                        param_name = param.spelling if param.spelling else "unnamed"
                        params.append((param_type, param_name))
                constructors.append(params)
                
        if constructors:
            print(f"    Found {len(constructors)} constructor(s) for {class_name}:")
            for i, params in enumerate(constructors):
                param_strs = [f"{ptype} {pname}" for ptype, pname in params]
                print(f"      Constructor {i+1}: {class_name}({', '.join(param_strs)})")
        else:
            print(f"    No explicit constructors found for {class_name} (default constructor available)")
            constructors.append([])  # Add default constructor
        
        return constructors
                
        

    # Helper: robustly determine whether `node` directly inherits from `base_name`.
    # Uses the CXX_BASE_SPECIFIER children and compares declaration spellings,
    # type spellings and displaynames. Strips template args and qualifiers
    # (namespace) and compares the simple name.
    def is_subclass_of(node, base_name):
        for b in node.get_children():
            if b.kind != clang.cindex.CursorKind.CXX_BASE_SPECIFIER: # type: ignore
                continue
            # Try to get the declaration spelling (best result)
            decl = None
            try:
                decl = b.type.get_declaration()
            except Exception:
                decl = None

            candidates = []
            if decl and decl.spelling:
                candidates.append(decl.spelling)
            tsp = getattr(b.type, 'spelling', None)
            if tsp:
                candidates.append(tsp)
            disp = getattr(b, 'displayname', None)
            if disp:
                candidates.append(disp)

            # Debugging: print candidates for provider-related classes to inspect AST
            # (no debug output)

            for name in candidates:
                if not name:
                    continue
                # remove template args (e.g. Foo<Bar>) and surrounding whitespace
                name = re.sub(r'<.*>$', '', name).strip()
                # compare both fully-qualified and simple name
                simple = name.split('::')[-1]
                if simple == base_name or name == base_name:
                    return True
        return False

    # Fallback: if libclang doesn't report base-specifiers (e.g. due to template
    # oddities or incomplete specializations), do a quick regex search in the
    # header text for a class declaration that lists the base by name.
    def text_inherits(header_path, node_name, base_name):
        try:
            with open(header_path, 'r') as fh:
                txt = fh.read()
        except Exception:
            return False
        # match `class NAME ... : ... BASE` (allow templates and qualifiers)
        pat = rf"class\s+{re.escape(node_name)}[^{{]*:\s*[^{{;]*\b{re.escape(base_name)}\b"
        return re.search(pat, txt, flags=re.MULTILINE | re.DOTALL) is not None

    print("Header files to scan:", headers)
    
    for h in headers:
        print("Parsing:", h)
        # Parse as C++17 with possibly extra include paths
        tu = index.parse(h, args=parse_args) # type: ignore
        for node in tu.cursor.walk_preorder():
            if node.kind in (clang.cindex.CursorKind.CLASS_DECL,  # type: ignore
                             clang.cindex.CursorKind.STRUCT_DECL,  # type: ignore
                             clang.cindex.CursorKind.CLASS_TEMPLATE):  # type: ignore
                # Check Vererbung & Abstraktion
                #print("Visiting:", node.spelling, "of kind", node.kind, "in", h)
                for base_class in base_classes:
                    is_containing, base_template_params = is_containing_class_declaration(node, base_class, h)
                    if is_containing:
                        print(f"File {h} contains class {base_class}")
                        base_classes_found[base_class] = h
                        template_parameters[base_class] = base_template_params
                        continue
                        
                    is_sub = is_subclass_of(node, base_class)
                    # fallback to a lightweight text-based check if AST didn't expose the
                    # base specifier (some template/incorrect declarations confuse libclang)
                    if not is_sub and node.spelling:
                        is_sub = text_inherits(h, node.spelling, base_class)

                    #print("  Is subclass of", base_class, ":", is_sub, "Abstract:", node.is_abstract_record() , "Spelling:", node.spelling, "in", h)
                    if is_sub and not node.is_abstract_record() and node.spelling:
                        if node.spelling != base_class:
                            # Get template parameters for this subclass
                            subclass_template_params = get_template_parameters(node)
                            if subclass_template_params:
                                print(f"    Subclass {node.spelling} template parameters: {subclass_template_params}")
                            
                            # Get constructors for this subclass
                            subclass_constructors_list = get_class_constructors(node, node.spelling)
                            subclass_constructors[base_class][node.spelling] = subclass_constructors_list
                            
                            # Store subclass with template info
                            found_classes[base_class].append((node.spelling, h, subclass_template_params))

    # Datei schreiben
    with open(output_path, "w") as f:
        
        f.write(f"#include <string>\n#include <vector>\n\n")
        
        for base_class in base_classes:
            if base_class in base_classes_found:
                f.write(f'#include "{base_classes_found[base_class]}" // {base_class} \n')
        
        for base_class in base_classes:
            f.write(f'\n// Includes for subclasses of {base_class}\n')
            for entry in found_classes[base_class]:
                # Handle both old format (cls, h) and new format (cls, h, template_params)
                if len(entry) == 3:
                    cls, h, _ = entry
                else:
                    cls, h = entry
                f.write(f'#include "{h}" // {cls} \n')
        
        f.write("\n")
        f.write("\n")
        f.write("\n")
                
        for base_class in base_classes:
            # Generate template parameters for the factory function
            base_template_params = template_parameters.get(base_class, ['In', 'Out'])  # fallback to In, Out
            template_decl = ", ".join([f"typename {param}" for param in base_template_params])
            template_args = ", ".join(base_template_params)
            
            template_str = f"template<{template_decl}>\n{base_class}<{template_args}>*"
            
            if len(base_template_params) == 0:
                template_str = f"{base_class}*"
            
            text = f"{template_str} create_instance_{base_class.lower()}(const std::string &class_name, std::vector<void*> parameter) {{\n"
            
            text += f"\n"
            
            
            
            count1 = 0
            count2 = 0
            for entry in found_classes[base_class]:
                # Handle both old format (cls, h) and new format (cls, h, template_params)
                if len(entry) == 3:
                    cls, h, subclass_template_params = entry
                else:
                    cls, h = entry
                    subclass_template_params = []
                    
                count2 += 1
                if cls in base_classes:
                    continue
                if cls not in subclass_constructors[base_class]:
                    continue
                count1 += 1
                if count1 == 1:
                    text += f'        if (class_name == "{cls}") {{\n'
                else:
                    text += f'        }} else if (class_name == "{cls}") {{\n'
                
                # Use template parameters for the subclass instantiation
                if subclass_template_params:
                    subclass_template_args = ", ".join(base_template_params[:len(subclass_template_params)])
                    cls_instantiation = f"{cls}<{subclass_template_args}>"
                else:
                    cls_instantiation = cls
                
                for ctor in subclass_constructors[base_class].get(cls, []):
                    param_count = len(ctor)
                    text += f'            // Constructor with {param_count} parameter(s)\n'
                    text += f'            // Parameters: {", ".join([ptype + " " + pname for ptype, pname in ctor])}\n'
                    text += f'            if (parameter.size() == {param_count}) {{\n'
                    text += f'                try {{\n'
                    text += f'                    return new {cls_instantiation}('
                    param_list = []
                    for i, (ptype, pname) in enumerate(ctor):
                        param_list.append(f'*reinterpret_cast<{ptype}*>(parameter[{i}])')
                    text += ', '.join(param_list)
                    text += f');\n'
                    text += f'                }} catch (...) {{\n'
                    text += f'                    // Handle constructor exceptions if necessary\n'
                    text += f'                }}\n'
                    text += f'            }}\n'
                text += f'            return nullptr;\n'
                if count2 == found_classes[base_class].__len__():
                    text += f'        }}\n'
                
            text += f"\n\n"
            
            # otherwise, return nullptr
            text += f"    return nullptr;\n"
            
            text += f"}}\n\n"
            f.write(text)
        

if __name__ == "__main__":
    generate()
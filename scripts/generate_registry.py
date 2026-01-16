#!/usr/bin/env python3
"""
Registry Generator Script

Parses C++ header files using libclang to find base classes and their subclasses,
then generates a registry header file with factory functions and lookup utilities.

Usage:
    python generate_registry.py <output_path> <base_classes> <header_files...>
"""

import sys
import os
import clang.cindex
import errno
import shlex
import re


# =============================================================================
# Libclang Initialization
# =============================================================================

def init_libclang():
    """Initialize libclang, trying multiple common paths if needed."""
    try:
        return clang.cindex.Index.create()
    except clang.cindex.LibclangError:
        candidates = [
            os.environ.get('LIBCLANG_PATH'),
            '/usr/lib/llvm-18/lib/libclang.so.1',
            '/usr/lib/x86_64-linux-gnu/libclang-18.so.18',
            '/usr/lib/x86_64-linux-gnu/libclang-18.so.1',
            '/usr/lib/llvm-15/lib/libclang.so.1',
            '/usr/lib/llvm-11/lib/libclang.so.1',
        ]
        for path in candidates:
            if not path or not os.path.exists(path):
                continue
            try:
                clang.cindex.Config.set_library_file(path)
                return clang.cindex.Index.create()
            except OSError as e:
                if getattr(e, 'errno', None) in (errno.ENOENT,):
                    continue
        raise


def get_parse_args():
    """Get clang parse arguments from environment or use defaults."""
    env_args = os.environ.get('CLANG_ARGS')
    if env_args:
        try:
            return shlex.split(env_args)
        except Exception:
            pass
    
    args = ['-std=c++17', '-I.', '-Iinclude', '-xc++']
    for sys_include in ['/usr/include/c++/11', '/usr/include/c++/13', '/usr/include']:
        if os.path.exists(sys_include):
            args.append(f'-I{sys_include}')
    return args


# =============================================================================
# AST Inspection Utilities
# =============================================================================

def get_template_parameters(node):
    """Extract template parameter names from a class template node."""
    template_params = []
    if node.kind != clang.cindex.CursorKind.CLASS_TEMPLATE: # type: ignore
        return template_params
    
    for child in node.get_children():
        if child.kind == clang.cindex.CursorKind.TEMPLATE_TYPE_PARAMETER: # type: ignore
            param_name = child.spelling or f"T{len(template_params)}"
            template_params.append(param_name)
        elif child.kind == clang.cindex.CursorKind.TEMPLATE_NON_TYPE_PARAMETER: # type: ignore
            param_name = child.spelling or f"N{len(template_params)}"
            template_params.append(param_name)
    return template_params


def get_class_metadata(node, header_path):
    """Extract @registry_name, @registry_aliases, and @category from comments above class."""
    if not node.location or not node.location.line:
        return {}
    
    try:
        with open(header_path, 'r') as f:
            lines = f.readlines()
        
        class_line = node.location.line - 1  # 0-indexed
        metadata = {}
        
        for i in range(max(0, class_line - 15), class_line):
            line = lines[i]
            if match := re.match(r'//\s*@registry_name:\s*(.+)', line):
                metadata['registry_name'] = match.group(1).strip()
            elif match := re.match(r'//\s*@registry_aliases:\s*(.+)', line):
                metadata['aliases'] = [a.strip() for a in match.group(1).split(',')]
            elif match := re.match(r'//\s*@category:\s*(.+)', line):
                metadata['category'] = match.group(1).strip()
        
        return metadata
    except Exception:
        return {}


def get_class_constructors(node, class_name):
    """
    Extract constructor information from a class node.
    Returns: list of constructors, each as [(param_type, param_name, default_value), ...]
    """
    constructors = []
    
    for child in node.get_children():
        if child.kind != clang.cindex.CursorKind.CONSTRUCTOR: # type: ignore
            continue
        
        params = []
        for param in child.get_children():
            if param.kind != clang.cindex.CursorKind.PARM_DECL: # type: ignore
                continue
            
            param_type = param.type.spelling if param.type else "unknown"
            param_name = param.spelling or "unnamed"
            default_value = _extract_default_value(param)
            params.append((param_type, param_name, default_value))
        
        constructors.append(params)
    
    if not constructors:
        constructors.append([])  # Default constructor
    
    _print_constructors(class_name, constructors)
    return constructors


def _extract_default_value(param_node):
    """Extract default value from a parameter declaration node."""
    default_expr_kinds = (
        clang.cindex.CursorKind.INTEGER_LITERAL, # type: ignore
        clang.cindex.CursorKind.FLOATING_LITERAL, # type: ignore
        clang.cindex.CursorKind.STRING_LITERAL, # type: ignore
        clang.cindex.CursorKind.CXX_BOOL_LITERAL_EXPR, # type: ignore
        clang.cindex.CursorKind.CXX_NULL_PTR_LITERAL_EXPR, # type: ignore
        clang.cindex.CursorKind.UNEXPOSED_EXPR, # type: ignore
        clang.cindex.CursorKind.CALL_EXPR, # type: ignore
        clang.cindex.CursorKind.UNARY_OPERATOR, # type: ignore
    )
    
    for child in param_node.get_children():
        if child.kind in default_expr_kinds:
            try:
                tokens = list(child.get_tokens())
                if tokens:
                    return ' '.join(t.spelling for t in tokens)
            except Exception:
                pass
            break
    return None


def _print_constructors(class_name, constructors):
    """Print constructor information for debugging."""
    if constructors and constructors != [[]]:
        print(f"    Found {len(constructors)} constructor(s) for {class_name}:")
        for i, params in enumerate(constructors):
            param_strs = []
            for ptype, pname, pdefault in params:
                if pdefault:
                    param_strs.append(f"{ptype} {pname} = {pdefault}")
                else:
                    param_strs.append(f"{ptype} {pname}")
            print(f"      Constructor {i+1}: {class_name}({', '.join(param_strs)})")
    else:
        print(f"    No explicit constructors found for {class_name} (default constructor available)")


# =============================================================================
# Inheritance Detection
# =============================================================================

def is_subclass_of(node, base_name):
    """Check if node directly inherits from base_name using AST."""
    for child in node.get_children():
        if child.kind != clang.cindex.CursorKind.CXX_BASE_SPECIFIER: # type: ignore
            continue
        
        candidates = []
        try:
            decl = child.type.get_declaration()
            if decl and decl.spelling:
                candidates.append(decl.spelling)
        except Exception:
            pass
        
        if tsp := getattr(child.type, 'spelling', None):
            candidates.append(tsp)
        if disp := getattr(child, 'displayname', None):
            candidates.append(disp)
        
        for name in candidates:
            if not name:
                continue
            name = re.sub(r'<.*>$', '', name).strip()
            simple = name.split('::')[-1]
            if simple == base_name or name == base_name:
                return True
    return False


def text_inherits(header_path, node_name, base_name):
    """Fallback: check inheritance via regex in source text."""
    try:
        with open(header_path, 'r') as f:
            txt = f.read()
    except Exception:
        return False
    
    pattern = rf"class\s+{re.escape(node_name)}[^{{]*:\s*[^{{;]*\b{re.escape(base_name)}\b"
    return re.search(pattern, txt, flags=re.MULTILINE | re.DOTALL) is not None


def is_base_class_declaration(node, class_name, current_file):
    """
    Check if node declares the base class in current file.
    Returns: (is_base_class, template_params)
    """
    if node.kind not in (
        clang.cindex.CursorKind.CLASS_DECL, # type: ignore
        clang.cindex.CursorKind.STRUCT_DECL, # type: ignore
        clang.cindex.CursorKind.CLASS_TEMPLATE, # type: ignore
    ):
        return False, []
    
    name = re.sub(r'<.*>$', '', node.spelling or "").strip()
    simple = name.split('::')[-1]
    
    if simple != class_name and name != class_name:
        return False, []
    
    node_file = str(node.location.file) if node.location.file else ""
    if not (node_file.endswith(current_file) or current_file.endswith(node_file)):
        return False, []
    
    print(f"  {node.spelling} (defined in {node_file})")
    template_params = get_template_parameters(node)
    if template_params:
        print(f"    Template parameters: {template_params}")
    
    return True, template_params


# =============================================================================
# Path Utilities
# =============================================================================

def normalize_include_path(path):
    """Convert absolute or relative path to relative include path."""
    if "include/" in path:
        return path[path.rfind("include/") + len("include/"):]
    if path.startswith("include/"):
        return path[len("include/"):]
    return path


# =============================================================================
# Code Generation
# =============================================================================

def write_includes(f, base_classes, base_classes_found, found_classes):
    """Generate #include statements."""
    f.write("#pragma once\n\n")
    f.write("#include <string>\n#include <vector>\n#include <unordered_map>\n\n")
    
    # Base class includes
    for base_class in base_classes:
        if base_class in base_classes_found:
            path = normalize_include_path(base_classes_found[base_class])
            f.write(f'#include "{path}" // {base_class} \n')
    
    # Subclass includes
    for base_class in base_classes:
        f.write(f'\n// Includes for subclasses of {base_class}\n')
        for entry in found_classes[base_class]:
            cls, h = entry[0], entry[1]
            path = normalize_include_path(h)
            f.write(f'#include "{path}" // {cls} \n')
    
    f.write("\n\n\n")


def write_lookup_functions(f, base_classes, base_class_categories, found_classes, class_metadata):
    """Generate name/alias lookup functions."""
    for base_class in base_classes:
        category = base_class_categories.get(base_class, base_class.lower())
        
        f.write(f"// Lookup function for {base_class} ({category})\n")
        f.write(f"// Maps registry names and aliases to actual class names\n")
        f.write(f"inline std::string resolve_{category}_class_name(const std::string& name_or_alias) {{\n")
        f.write(f"    static const std::unordered_map<std::string, std::string> lookup = {{\n")
        
        for entry in found_classes[base_class]:
            cls = entry[0]
            if cls in base_classes:
                continue
            
            metadata = class_metadata[base_class].get(cls, {})
            if 'registry_name' in metadata:
                f.write(f'        {{"{metadata["registry_name"]}", "{cls}"}},\n')
            for alias in metadata.get('aliases', []):
                f.write(f'        {{"{alias}", "{cls}"}},\n')
        
        f.write("    };\n\n")
        f.write("    auto it = lookup.find(name_or_alias);\n")
        f.write("    if (it != lookup.end()) {\n")
        f.write("        return it->second;\n")
        f.write("    }\n")
        f.write("    return name_or_alias; // Return as-is if no mapping found\n")
        f.write("}\n\n")


def write_category_lookup(f, base_classes, base_class_categories):
    """Generate category to base class lookup function."""
    f.write("// Lookup function to resolve category names to base class names\n")
    f.write("inline std::string resolve_category_to_base_class(const std::string& category) {\n")
    f.write("    static const std::unordered_map<std::string, std::string> lookup = {\n")
    
    for base_class in base_classes:
        category = base_class_categories.get(base_class, base_class.lower())
        f.write(f'        {{"{category}", "{base_class}"}},\n')
    
    f.write("    };\n\n")
    f.write("    auto it = lookup.find(category);\n")
    f.write("    if (it != lookup.end()) {\n")
    f.write("        return it->second;\n")
    f.write("    }\n")
    f.write("    return category; // Return as-is if no mapping found\n")
    f.write("}\n\n")


def _get_template_strings(base_class, template_params):
    """Get template declaration and argument strings."""
    if not template_params:
        return f"{base_class}*", ""
    
    template_decl = ", ".join(f"typename {p}" for p in template_params)
    template_args = ", ".join(template_params)
    return f"template<{template_decl}>\n{base_class}<{template_args}>*", template_args


def _get_class_instantiation(cls, subclass_template_params, base_template_params):
    """Get instantiation string for a class (with template args if needed)."""
    if subclass_template_params:
        args = ", ".join(base_template_params[:len(subclass_template_params)])
        return f"{cls}<{args}>"
    return cls


def write_factory_functions(f, base_classes, template_parameters, base_class_categories,
                            found_classes, subclass_constructors):
    """Generate factory functions for creating instances."""
    for base_class in base_classes:
        base_template_params = template_parameters.get(base_class, ['In', 'Out'])
        template_str, template_args = _get_template_strings(base_class, base_template_params)
        category = base_class_categories.get(base_class, base_class.lower())
        
        # Filter valid entries
        entries = [(cls, h, tparams) for cls, h, tparams in found_classes[base_class]
                   if cls not in base_classes and cls in subclass_constructors[base_class]]
        
        # Vector-based factory
        _write_vector_factory(f, base_class, template_str, base_template_params,
                              entries, subclass_constructors)
        
        # Map-based factory with name resolution
        _write_map_factory(f, base_class, template_str, template_args, category,
                           base_template_params, entries, subclass_constructors)


def _write_vector_factory(f, base_class, template_str, base_template_params,
                          entries, subclass_constructors):
    """Generate vector<void*> parameter factory."""
    f.write(f"{template_str} create_instance_{base_class.lower()}(const std::string &class_name, std::vector<void*> parameter) {{\n\n")
    
    for i, (cls, _, subclass_template_params) in enumerate(entries):
        if i == 0:
            f.write(f'    if (class_name == "{cls}") {{\n')
        else:
            f.write(f'    }} else if (class_name == "{cls}") {{\n')
        
        cls_inst = _get_class_instantiation(cls, subclass_template_params, base_template_params)
        
        for ctor in subclass_constructors[base_class].get(cls, []):
            param_count = len(ctor)
            param_docs = [f"{t} {n} = {d}" if d else f"{t} {n}" for t, n, d in ctor]
            
            f.write(f'        // Constructor with {param_count} parameter(s)\n')
            f.write(f'        // Parameters: {", ".join(param_docs)}\n')
            f.write(f'        if (parameter.size() == {param_count}) {{\n')
            f.write(f'            try {{\n')
            
            params = ', '.join(f'*reinterpret_cast<{t}*>(parameter[{i}])' for i, (t, _, _) in enumerate(ctor))
            f.write(f'                return new {cls_inst}({params});\n')
            
            f.write(f'            }} catch (...) {{\n')
            f.write(f'                // Handle constructor exceptions if necessary\n')
            f.write(f'            }}\n')
            f.write(f'        }}\n')
        
        f.write(f'        return nullptr;\n')
    
    if entries:
        f.write("    }\n")
    
    f.write("\n    return nullptr;\n}\n\n")


def _write_map_factory(f, base_class, template_str, template_args, category,
                       base_template_params, entries, subclass_constructors):
    """Generate unordered_map<string, void*> parameter factory with name resolution."""
    f.write(f"{template_str} create_instance_{base_class.lower()}(const std::string &class_name, std::unordered_map<std::string,void*> parameter) {{\n")
    f.write(f"    // Resolve name or alias to actual class name\n")
    f.write(f"    std::string resolved_class_name = resolve_{category}_class_name(class_name);\n\n")
    
    for i, (cls, _, subclass_template_params) in enumerate(entries):
        if i == 0:
            f.write(f'    if (resolved_class_name == "{cls}") {{\n')
        else:
            f.write(f'    }} else if (resolved_class_name == "{cls}") {{\n')
        
        for ctor in subclass_constructors[base_class].get(cls, []):
            param_count = len(ctor)
            params_with_defaults = sum(1 for _, _, d in ctor if d is not None)
            params_without_defaults = param_count - params_with_defaults
            
            param_docs = [f"{t} {n} = {d}" if d else f"{t} {n}" for t, n, d in ctor]
            
            f.write(f'        // Constructor with {param_count} parameter(s)\n')
            f.write(f'        // Parameters: {", ".join(param_docs)}\n')
            
            if params_with_defaults > 0:
                f.write(f'        if (parameter.size() >= {params_without_defaults} && parameter.size() <= {param_count}) {{\n')
            else:
                f.write(f'        if (parameter.size() == {param_count}) {{\n')
            
            f.write(f'            std::vector<void*> params_vector;\n')
            f.write(f'            try {{\n')
            f.write(f'                // Extract parameters from the map\n')
            
            for ptype, pname, pdefault in ctor:
                if pdefault:
                    f.write(f'                if (parameter.find("{pname}") != parameter.end()) {{\n')
                    f.write(f'                    params_vector.push_back(parameter.at("{pname}"));\n')
                    f.write(f'                }} else {{\n')
                    f.write(f'                    static {ptype} default_{pname} = {pdefault};\n')
                    f.write(f'                    params_vector.push_back(&default_{pname});\n')
                    f.write(f'                }}\n')
                else:
                    f.write(f'                params_vector.push_back(parameter.at("{pname}"));\n')
            
            template_call = f"<{template_args}>" if template_args else ""
            f.write(f'                return create_instance_{base_class.lower()}{template_call}(class_name, params_vector);\n')
            f.write(f'            }} catch (...) {{\n')
            f.write(f'                // Handle exceptions if necessary\n')
            f.write(f'            }}\n')
            f.write(f'        }}\n')
        
        f.write(f'        return nullptr;\n')
    
    if entries:
        f.write("    }\n")
    
    f.write("    return nullptr;\n}\n\n")


# =============================================================================
# Main Entry Point
# =============================================================================

def generate():
    if len(sys.argv) < 4:
        print("Usage: generate_registry.py <output_path> <base_classes> <header_files...>")
        sys.exit(1)
    
    output_path = sys.argv[1]
    base_class = sys.argv[2]
    headers = sys.argv[3:]
    
    base_classes = base_class.split(",") if "," in base_class else [base_class]

    # Initialize libclang
    index = init_libclang()
    parse_args = get_parse_args()
    
    # Data structures
    found_classes = {bc: [] for bc in base_classes}
    base_classes_found = {}
    subclass_constructors = {bc: {} for bc in base_classes}
    template_parameters = {bc: [] for bc in base_classes}
    class_metadata = {bc: {} for bc in base_classes}
    base_class_categories = {}

    print("Header files to scan:", headers)
    
    for h in headers:
        print("Parsing:", h)
        tu = index.parse(h, args=parse_args)
        for node in tu.cursor.walk_preorder():
            if node.kind not in (clang.cindex.CursorKind.CLASS_DECL, # type: ignore
                                 clang.cindex.CursorKind.STRUCT_DECL, # type: ignore
                                 clang.cindex.CursorKind.CLASS_TEMPLATE): # type: ignore
                continue
            
            for base_class in base_classes:
                is_base, base_template_params = is_base_class_declaration(node, base_class, h)
                if is_base:
                    print(f"File {h} contains class {base_class}")
                    base_classes_found[base_class] = h
                    template_parameters[base_class] = base_template_params
                    
                    # Extract category metadata for base class
                    base_metadata = get_class_metadata(node, h)
                    if 'category' in base_metadata:
                        base_class_categories[base_class] = base_metadata['category']
                        print(f"    Base class category: {base_metadata['category']}")
                    
                    continue
                
                is_sub = is_subclass_of(node, base_class)
                # Fallback to text-based check if AST didn't expose base specifier
                if not is_sub and node.spelling:
                    is_sub = text_inherits(h, node.spelling, base_class)

                if is_sub and not node.is_abstract_record() and node.spelling:
                    if node.spelling != base_class:
                        # Only process classes defined in the current header file (not included files)
                        node_file = str(node.location.file) if node.location.file else ""
                        if not (node_file.endswith(h) or h.endswith(node_file)):
                            continue
                        
                        # Get template parameters for this subclass
                        subclass_template_params = get_template_parameters(node)
                        if subclass_template_params:
                            print(f"    Subclass {node.spelling} template parameters: {subclass_template_params}")
                        
                        # Get metadata for this subclass
                        metadata = get_class_metadata(node, h)
                        if metadata:
                            class_metadata[base_class][node.spelling] = metadata
                            if 'registry_name' in metadata:
                                print(f"    Registry name: {metadata['registry_name']}")
                            if 'aliases' in metadata:
                                print(f"    Aliases: {', '.join(metadata['aliases'])}")
                        
                        # Get constructors for this subclass
                        subclass_constructors_list = get_class_constructors(node, node.spelling)
                        subclass_constructors[base_class][node.spelling] = subclass_constructors_list
                        
                        # Store subclass with template info
                        found_classes[base_class].append((node.spelling, h, subclass_template_params))

    # Write output file
    with open(output_path, "w") as f:
        write_includes(f, base_classes, base_classes_found, found_classes)
        write_lookup_functions(f, base_classes, base_class_categories, found_classes, class_metadata)
        write_category_lookup(f, base_classes, base_class_categories)
        write_factory_functions(f, base_classes, template_parameters, base_class_categories,
                                found_classes, subclass_constructors)


if __name__ == "__main__":
    generate()
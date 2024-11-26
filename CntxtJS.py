#CntxtJS.py - JavaScript/TypeScript codebase analyzer that generates comprehensive knowledge graphs optimized for LLM context windows

import os
import re
import sys
import json
import networkx as nx
from networkx.readwrite import json_graph
from typing import Dict, List, Optional, Set, Any


class JSCodeKnowledgeGraph:
    def __init__(self, directory: str):
        """Initialize the knowledge graph generator.

        Args:
            directory: Root directory of the JavaScript/TypeScript codebase.
        """
        self.directory = directory
        self.graph = nx.DiGraph()
        self.class_methods: Dict[str, Set[str]] = {}
        self.function_params: Dict[str, List[Dict[str, Any]]] = {}
        self.function_returns: Dict[str, str] = {}
        self.files_processed = 0
        self.total_files = 0
        self.dirs_processed = 0

        # Add alias mapping support.
        self.alias_map = {
            "@/": "src/",
            "@components/": "components/",
            "@lib/": "lib/",
            "@utils/": "utils/",
            "@hooks/": "hooks/",
            "@contexts/": "contexts/",
            "@types/": "types/",
            "@app/": "app/",
        }

        # Track analyzed files to prevent circular dependencies.
        self.analyzed_files = set()

        # Map exported entities to their defining files.
        self.exports_map: Dict[str, Set[str]] = {}

        # Directories to ignore during analysis.
        self.ignored_directories = set([
            'node_modules', 'build', 'dist', 'public', 'static', 'types', '.env', '.cache',
            'cache', '.next', 'coverage', '.results', 'results', 'screenshots', 'videos',
            'tmp', 'temp', 'logs', 'out', 'aot', '.nuxt', 'migrations', 'static',
            'wwwroot', '.meteor', 'local', 'reports', 'docs', 'config', '.config', '.vscode',
            '.idea', '.git'
        ])

        # Files to ignore during analysis.
        self.ignored_files = set([
            '.gitignore',
            '.env',
        ])

        # For processing dependencies
        self.dependencies: Dict[str, Set[str]] = {}

        # Counters for statistics
        self.total_classes = 0
        self.total_functions = 0
        self.total_components = 0
        self.total_hooks = 0
        self.total_dependencies = set()
        self.total_imports = 0
        self.total_exports = 0
        self._discovered_components = set()

    def analyze_codebase(self):
        """Analyze the JavaScript/TypeScript codebase to extract files, imports,
        classes, methods, and their relationships."""
        # First pass to count total files
        print("\nCounting files...")
        for root, dirs, files in os.walk(self.directory):
            # Remove ignored directories from dirs in-place to prevent walking into them
            dirs[:] = [d for d in dirs if d not in self.ignored_directories]

            # Skip if current directory is in node_modules or other ignored directories
            if any(ignored in root.split(os.sep) for ignored in self.ignored_directories):
                continue
            dirs[:] = [d for d in dirs if d not in self.ignored_directories]
            self.total_files += sum(1 for f in files if f not in self.ignored_files and f.endswith((".js", ".ts", ".jsx", ".tsx", ".d.ts")))

        print(f"Found {self.total_files} JavaScript/TypeScript files to process")
        print("\nProcessing files...")

        # Second pass to process files
        for root, dirs, files in os.walk(self.directory):
            # Remove ignored directories from dirs in-place to prevent walking into them
            dirs[:] = [d for d in dirs if d not in self.ignored_directories]

            # Skip if current directory is in node_modules or other ignored directories
            if any(ignored in root.split(os.sep) for ignored in self.ignored_directories):
                continue

            # Display current directory
            rel_path = os.path.relpath(root, self.directory)
            self.dirs_processed += 1
            print(f"\rProcessing directory [{self.dirs_processed}]: {rel_path}", end="")

            for file in files:
                if file in self.ignored_files:
                    continue
                if file.endswith((".js", ".ts", ".jsx", ".tsx", ".d.ts")):
                    file_path = os.path.join(root, file)
                    self._process_file(file_path)
                elif file in ["package.json", "package-lock.json", "yarn.lock", "pnpm-lock.yaml"]:
                    file_path = os.path.join(root, file)
                    self._process_dependency_file(file_path)

        print(f"\n\nCompleted processing {self.files_processed} files across {self.dirs_processed} directories")

    def _process_file(self, file_path: str):
        """Process a file to detect imports, classes, methods, and functions."""
        if file_path in self.analyzed_files:
            return

        try:
            self.files_processed += 1
            relative_path = os.path.relpath(file_path, self.directory)
            print(f"\rProcessing file [{self.files_processed}/{self.total_files}]: {relative_path}", end="", flush=True)

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            relative_path = os.path.relpath(file_path, self.directory)
            file_node = f"File: {relative_path}"

            # Add to analyzed files set.
            self.analyzed_files.add(file_path)

            # Add file node if it doesn't exist.
            if not self.graph.has_node(file_node):
                self.graph.add_node(file_node, type="file", path=relative_path)

            # Process the file contents.
            self._process_exports(content, file_node)
            self._process_imports(content, file_node)
            self._process_classes(content, file_node)
            self._process_functions(content, file_node)
            self._process_jsx_components(content, file_node)
            self._process_hooks(content, file_node)

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}", file=sys.stderr)

    def _process_imports(self, content: str, file_node: str):
        """Process import statements in the content."""
        import_patterns = [
            # Destructured imports.
            r'import\s*{([^}]*)}\s*from\s*[\'"]([^\'"]+)[\'"]',
            # Default imports with optional destructuring.
            r'import\s+(?:type\s+)?(\w+)\s*(?:,\s*{([^}]*)})?(?:\s*,\s*\*\s+as\s+\w+)?\s*from\s*[\'"]([^\'"]+)[\'"]',
            # Namespace imports.
            r'import\s*\*\s+as\s+(\w+)\s+from\s*[\'"]([^\'"]+)[\'"]',
            # Side effect imports.
            r'import\s*[\'"]([^\'"]+)[\'"]',
            # Dynamic imports.
            r'import\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            # Require statements.
            r'(?:const|let|var)?\s*(?:{[^}]*})?\s*(?:[\w\s,{]*)\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
            # Type imports.
            r'import\s+type\s*{([^}]*)}\s*from\s*[\'"]([^\'"]+)[\'"]',
            # Re-exports.
            r'export\s*(?:\*|{[^}]*})\s*from\s*[\'"]([^\'"]+)[\'"]',
            # Export equals.
            r'export\s*=\s*require\s*\(\s*[\'"]([^\'"]+)[\'"]\s*\)',
        ]

        # Handle multi-line imports.
        lines = content.split('\n')
        current_import = ""
        all_imports = []

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if ('import' in line or 'require' in line or 'export' in line) and 'from' in line:
                if (
                    line.count('{') != line.count('}')
                    or not any(c in line for c in [';', ')', '}'])
                    or line.count('(') != line.count(')')
                ):
                    current_import = line
                    continue
                else:
                    line_to_process = line
            elif current_import:
                current_import += " " + line
                if (
                    current_import.count('{') == current_import.count('}')
                    and any(c in current_import for c in [';', ')', '}'])
                    and current_import.count('(') == current_import.count(')')
                ):
                    line_to_process = current_import
                    current_import = ""
                else:
                    continue
            else:
                continue

            # Process the complete import statement.
            for pattern in import_patterns:
                matches = re.finditer(pattern, line_to_process, re.MULTILINE)
                for match in matches:
                    groups = match.groups()
                    if not groups:
                        continue

                    import_entities = []
                    # Get the import path (last group for most patterns).
                    import_path = groups[-1]

                    # Handle destructured imports.
                    if "{" in line_to_process:
                        destructured = next((g for g in groups if g and "{" in g), "")
                        if destructured:
                            # Process each destructured import.
                            imports = re.findall(r'(\w+)(?:\s+as\s+\w+)?', destructured)
                            import_entities.extend(imports)
                    else:
                        if groups[0]:
                            import_entities.append(groups[0])

                    all_imports.append((import_entities, import_path))

        # Process all found imports.
        for import_entities, imp in all_imports:
            if not imp:
                continue

            imported_file = self._resolve_import_path(file_node, imp)
            if imported_file:
                if not self.graph.has_node(imported_file):
                    self.graph.add_node(imported_file, type="file")
                self.graph.add_edge(file_node, imported_file, relation="IMPORTS")

                # Create nodes for imported entities and link them.
                for entity in import_entities:
                    entity_node = f"Entity: {entity}"
                    if not self.graph.has_node(entity_node):
                        self.graph.add_node(entity_node, type="entity", name=entity)
                    self.graph.add_edge(file_node, entity_node, relation="USES_ENTITY")
                    # If the entity is exported from the imported file, link it.
                    if entity in self.exports_map.get(imported_file, set()):
                        self.graph.add_edge(entity_node, imported_file, relation="DEFINED_IN")

                # Update import count.
                self.total_imports += 1

                # Track dependencies.
                self.total_dependencies.add(imp)

    def _resolve_import_path(self, current_file: str, import_path: str) -> Optional[str]:
        """Resolve the import path to an actual file."""
        try:
            # Remove 'File: ' prefix if present.
            if current_file.startswith("File: "):
                current_file = current_file[6:]

            # Handle absolute paths with aliases.
            for alias, replacement in self.alias_map.items():
                if import_path.startswith(alias):
                    import_path = import_path.replace(alias, replacement)
                    base_dir = self.directory
                    resolved_path = os.path.normpath(os.path.join(base_dir, import_path))
                    break
            else:
                if import_path.startswith((".", "/")):
                    # Handle relative paths.
                    current_dir = os.path.dirname(os.path.join(self.directory, current_file))
                    resolved_path = os.path.normpath(os.path.join(current_dir, import_path))
                else:
                    # Handle node_modules imports.
                    return f"External: {import_path}"

            # Try different extensions and index files.
            extensions = [
                "",
                ".js",
                ".ts",
                ".jsx",
                ".tsx",
                ".mjs",
                ".mts",
                "/index.js",
                "/index.ts",
                "/index.jsx",
                "/index.tsx",
                "/index.mjs",
                "/index.mts",
                ".d.ts",
            ]

            # First try exact path.
            for ext in extensions:
                full_path = resolved_path + ext
                if os.path.isfile(full_path):
                    relative_path = os.path.relpath(full_path, self.directory)
                    if self._is_in_ignored_directory(relative_path):
                        return None
                    return f"File: {relative_path}"

            # Try parent directory index files.
            parent_dir = os.path.dirname(resolved_path)
            for ext in extensions:
                if ext.startswith("/"):
                    full_path = parent_dir + ext
                    if os.path.isfile(full_path):
                        relative_path = os.path.relpath(full_path, self.directory)
                        if self._is_in_ignored_directory(relative_path):
                            return None
                        return f"File: {relative_path}"

            # Try directory index.
            if os.path.isdir(resolved_path):
                for ext in extensions:
                    if ext.startswith("/"):
                        full_path = resolved_path + ext
                        if os.path.isfile(full_path):
                            relative_path = os.path.relpath(full_path, self.directory)
                            if self._is_in_ignored_directory(relative_path):
                                return None
                            return f"File: {relative_path}"

            return None

        except Exception as e:
            print(f"Error resolving import path {import_path}: {str(e)}", file=sys.stderr)
            return None

    def _is_in_ignored_directory(self, relative_path: str) -> bool:
        """Check if the relative path is inside any ignored directory."""
        path_parts = relative_path.split(os.sep)
        for part in path_parts:
            if part in self.ignored_directories:
                return True
        return False

    def _process_classes(self, content: str, file_node: str):
        """Process class declarations including React components and interfaces."""
        class_patterns = [
            # Regular classes.
            r'(?:export\s+)?(?:abstract\s+)?class\s+(\w+)(?:\s+extends\s+[^{]+)?\s*{([^}]*)}',
            # Interfaces.
            r'(?:export\s+)?interface\s+(\w+)(?:\s+extends\s+[^{]+)?\s*{([^}]*)}',
            # Type aliases.
            r'(?:export\s+)?type\s+(\w+)\s*=\s*{([^}]*)}',
            # React components as classes.
            r'(?:export\s+)?class\s+(\w+)\s+extends\s+(?:React\.)?Component\s*[^{]*{([^}]*)}',
            # Pure components.
            r'(?:export\s+)?class\s+(\w+)\s+extends\s+(?:React\.)?PureComponent\s*[^{]*{([^}]*)}',
        ]

        for pattern in class_patterns:
            for match in re.finditer(pattern, content, re.DOTALL):
                try:
                    class_name = match.group(1)
                    class_node = f"Class: {class_name} ({file_node})"

                    if not self.graph.has_node(class_node):
                        self.graph.add_node(
                            class_node,
                            type="class",
                            name=class_name,
                            is_react_component="Component" in pattern or "PureComponent" in pattern,
                        )

                    self.graph.add_edge(file_node, class_node, relation="DEFINES")

                    class_body = match.group(2) if len(match.groups()) > 1 else ""
                    if class_body:
                        self._process_class_methods(class_body, class_node)

                    # If exported, add to exports map.
                    if 'export' in match.group(0):
                        if file_node not in self.exports_map:
                            self.exports_map[file_node] = set()
                        self.exports_map[file_node].add(class_name)

                    self.total_classes += 1

                except Exception as e:
                    print(f"Error processing class {class_name}: {str(e)}", file=sys.stderr)

    def _process_class_methods(self, class_body: str, class_node: str):
        """Process methods within a class including React lifecycle methods."""
        method_pattern = (
            r'(?:async\s+)?'                           # async modifier
            r'(?:static\s+)?'                          # static modifier
            r'(?:private\s+|protected\s+|public\s+)?'  # access modifiers
            r'(?:get\s+|set\s+)?'                      # getter/setter
            r'(\w+)'                                   # method name
            r'\s*'
            r'(?:<[^>]*>)?'                            # generic type parameters
            r'\s*'
            r'\((.*?)\)'                               # parameters
            r'(?:\s*:\s*([^{;]*))?'                    # return type
        )

        lifecycle_methods = {
            'componentDidMount', 'componentDidUpdate', 'componentWillUnmount',
            'shouldComponentUpdate', 'getSnapshotBeforeUpdate', 'componentDidCatch',
            'getDerivedStateFromProps', 'getDerivedStateFromError', 'render',
        }

        for match in re.finditer(method_pattern, class_body):
            try:
                method_name = match.group(1)
                parameters = match.group(2).strip() if match.group(2) else ""
                return_type = match.group(3).strip() if match.group(3) else None

                method_node = f"Function: {method_name} ({class_node})"

                if not self.graph.has_node(method_node):
                    self.graph.add_node(
                        method_node,
                        type="function",
                        name=method_name,
                        parameters=self._parse_parameters(parameters),
                        return_type=return_type,
                        is_lifecycle_method=method_name in lifecycle_methods,
                    )

                self.graph.add_edge(class_node, method_node, relation="HAS_FUNCTION")

                # Track class methods.
                if class_node not in self.class_methods:
                    self.class_methods[class_node] = set()
                self.class_methods[class_node].add(method_name)

                # Track function parameters and returns.
                self.function_params[method_node] = self._parse_parameters(parameters)
                if return_type:
                    self.function_returns[method_node] = return_type

                self.total_functions += 1

            except Exception as e:
                print(f"Error processing method {method_name}: {str(e)}", file=sys.stderr)

    def _process_functions(self, content: str, file_node: str):
        """Process standalone functions including React hooks and components."""
        function_patterns = [
            # Regular functions.
            r'(?:export\s+)?(?:async\s+)?function\s*(?:<[^>]*>)?\s*(\w+)\s*\((.*?)\)(?:\s*:\s*([^{;]*))?',
            # Arrow functions with explicit type.
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*:\s*(?:React\.)?(?:FC|FunctionComponent|ComponentType)[^=]*=\s*(?:async\s*)?\((.*?)\)(?:\s*:\s*([^{;]*))?',
            # Arrow functions.
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\((.*?)\)(?:\s*:\s*([^{;]*))?\s*=>',
            # React components.
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*React\.memo\(',
            # React forwardRef.
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*React\.forwardRef\(',
            # Custom hooks.
            r'(?:export\s+)?(?:function|const|let|var)\s+(use\w+)',
            # Higher-order components.
            r'(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*with\w+\(',
        ]

        for pattern in function_patterns:
            matches = re.finditer(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    func_name = match.group(1)

                    # Skip if this is a method call rather than a definition.
                    if re.search(r'\.\s*' + func_name + r'\s*\(', content):
                        continue

                    # Get parameters and return type if available.
                    parameters = match.group(2).strip() if len(match.groups()) > 1 and match.group(2) else ""
                    return_type = match.group(3).strip() if len(match.groups()) > 2 and match.group(3) else None

                    func_node = f"Function: {func_name} ({file_node})"

                    # Determine function type.
                    is_hook = func_name.startswith('use')
                    is_component = any(
                        suffix in func_name for suffix in ['Page', 'Component', 'View', 'Layout']
                    )

                    if not self.graph.has_node(func_node):
                        self.graph.add_node(
                            func_node,
                            type="function",
                            name=func_name,
                            parameters=self._parse_parameters(parameters),
                            return_type=return_type,
                            is_hook=is_hook,
                            is_component=is_component,
                        )

                    self.graph.add_edge(file_node, func_node, relation="DEFINES")

                    # Track parameters and return types.
                    self.function_params[func_node] = self._parse_parameters(parameters)
                    if return_type:
                        self.function_returns[func_node] = return_type

                    # If exported, add to exports map.
                    if 'export' in match.group(0):
                        if file_node not in self.exports_map:
                            self.exports_map[file_node] = set()
                        self.exports_map[file_node].add(func_name)

                    if is_component:
                        self.total_components += 1
                    elif is_hook:
                        self.total_hooks += 1
                    else:
                        self.total_functions += 1

                except Exception as e:
                    print(f"Error processing function {func_name}: {str(e)}", file=sys.stderr)

    def _process_jsx_components(self, content: str, file_node: str):
        """Process JSX/TSX component usage within files."""
        # Pattern to match JSX component usage.
        component_pattern = r'<([A-Z]\w+)(?:\s+(?:{[^}]*}|"[^"]*"|\'[^\']*\'|[^>])*)?/?>'

        for match in re.finditer(component_pattern, content):
            try:
                component_name = match.group(1)
                component_node = f"Component: {component_name}"

                if not self.graph.has_node(component_node):
                    self.graph.add_node(
                        component_node,
                        type="component",
                        name=component_name,
                    )

                self.graph.add_edge(file_node, component_node, relation="USES_COMPONENT")

            except Exception as e:
                print(f"Error processing JSX component {component_name}: {str(e)}", file=sys.stderr)

    def _process_hooks(self, content: str, file_node: str):
        """Process React hook usage within components."""
        hook_pattern = r'(use\w+)\s*\('

        for match in re.finditer(hook_pattern, content):
            try:
                hook_name = match.group(1)
                hook_node = f"Hook: {hook_name}"

                if not self.graph.has_node(hook_node):
                    self.graph.add_node(
                        hook_node,
                        type="hook",
                        name=hook_name,
                    )

                self.graph.add_edge(file_node, hook_node, relation="USES_HOOK")

            except Exception as e:
                print(f"Error processing hook {hook_name}: {str(e)}", file=sys.stderr)

    def _parse_parameters(self, params_str: str) -> List[Dict[str, Any]]:
        """Parse function parameters including TypeScript types and destructuring."""
        if not params_str:
            return []

        params = []
        depth = 0
        current_param = ""

        for char in params_str:
            if char in '{[(':
                depth += 1
            elif char in '}])':
                depth -= 1
            elif char == ',' and depth == 0:
                if current_param.strip():
                    params.append(self._parse_single_parameter(current_param.strip()))
                current_param = ""
                continue
            current_param += char

        if current_param.strip():
            params.append(self._parse_single_parameter(current_param.strip()))

        return params

    def _parse_single_parameter(self, param: str) -> Dict[str, Any]:
        """Parse a single parameter with its type and default value."""
        param_dict: Dict[str, Any] = {"name": param}

        # Handle TypeScript type annotations.
        type_match = re.match(r'(?:readonly\s+)?(\w+)\s*(?:\?|!)?:\s*([^=]+)', param)
        if type_match:
            param_dict["name"] = type_match.group(1)
            param_dict["type"] = type_match.group(2).strip()

        # Handle default values.
        default_match = re.match(r'(\w+)\s*=\s*(.+)', param)
        if default_match:
            param_dict["name"] = default_match.group(1)
            param_dict["default"] = default_match.group(2)

        # Handle destructuring.
        if param.startswith('{') or param.startswith('['):
            param_dict["destructured"] = True

        return param_dict

    def _process_exports(self, content: str, file_node: str):
        """Process export statements including default and named exports."""
        export_patterns = [
            # Named exports.
            r'export\s+(?:const|let|var|function|class)\s+(\w+)',
            # Default exports.
            r'export\s+default\s+(?:class\s+)?(\w+)',
            # Named exports list.
            r'export\s*{\s*((?:\w+(?:\s+as\s+\w+)?(?:\s*,\s*)?)+)\s*}',
            # Re-exports.
            r'export\s*\*\s*from\s*[\'"]([^\'"]+)[\'"]',
            # Type exports.
            r'export\s+type\s+(\w+)',
            # Export functions (e.g., export function funcName() {})
            r'export\s+function\s+(\w+)\s*\(',
        ]

        for pattern in export_patterns:
            for match in re.finditer(pattern, content):
                try:
                    if 'from' in pattern:
                        # Handle re-exports.
                        module_path = match.group(1)
                        # Resolve module path to file.
                        exported_file = self._resolve_import_path(file_node, module_path)
                        if exported_file:
                            if not self.graph.has_node(exported_file):
                                self.graph.add_node(exported_file, type="file")
                            self.graph.add_edge(file_node, exported_file, relation="RE_EXPORTS")
                    else:
                        exports = match.group(1).split(',')
                        for export in exports:
                            export_name = export.strip().split(' as ')[0].strip()
                            export_node = f"Entity: {export_name}"

                            if not self.graph.has_node(export_node):
                                self.graph.add_node(
                                    export_node,
                                    type="export",
                                    name=export_name,
                                )

                            self.graph.add_edge(file_node, export_node, relation="EXPORTS")

                            # Add to exports map.
                            if file_node not in self.exports_map:
                                self.exports_map[file_node] = set()
                            self.exports_map[file_node].add(export_name)

                            self.total_exports += 1

                except Exception as e:
                    print(f"Error processing export: {str(e)}", file=sys.stderr)

    def _process_dependency_file(self, file_path: str):
        """Process dependency files like package.json, package-lock.json, yarn.lock, pnpm-lock.yaml."""
        try:
            if file_path in self.analyzed_files:
                return

            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            relative_path = os.path.relpath(file_path, self.directory)
            file_node = f"Dependency File: {relative_path}"

            # Add to analyzed files set.
            self.analyzed_files.add(file_path)

            # Add file node if it doesn't exist.
            if not self.graph.has_node(file_node):
                self.graph.add_node(file_node, type="dependency_file", path=relative_path)

            # Process dependencies
            if file_path.endswith("package.json"):
                data = json.loads(content)
                dependencies = data.get("dependencies", {})
                dev_dependencies = data.get("devDependencies", {})

                for dep in {**dependencies, **dev_dependencies}:
                    dep_node = f"Dependency: {dep}"
                    if not self.graph.has_node(dep_node):
                        self.graph.add_node(dep_node, type="dependency", name=dep)
                    self.graph.add_edge(file_node, dep_node, relation="HAS_DEPENDENCY")

                    self.total_dependencies.add(dep)

            elif file_path.endswith(("package-lock.json", "yarn.lock", "pnpm-lock.yaml")):
                # For lock files, parse package-lock.json
                if file_path.endswith("package-lock.json"):
                    data = json.loads(content)
                    dependencies = data.get("dependencies", {})
                    for dep in dependencies:
                        dep_node = f"Dependency: {dep}"
                        if not self.graph.has_node(dep_node):
                            self.graph.add_node(dep_node, type="dependency", name=dep)
                        self.graph.add_edge(file_node, dep_node, relation="HAS_LOCKED_DEPENDENCY")

                        self.total_dependencies.add(dep)
                else:
                    # For yarn.lock and pnpm-lock.yaml, parsing is complex; skipping for now.
                    pass

        except Exception as e:
            print(f"Error processing dependency file {file_path}: {str(e)}", file=sys.stderr)

    def save_graph(self, output_path: str):
        """Save the knowledge graph in standard JSON format."""
        data = json_graph.node_link_data(self.graph, edges="links")
        metadata = {
            "stats": {
                "total_files": self.total_files,
                "total_classes": self.total_classes,
                "total_functions": self.total_functions,
                "total_exports": self.total_exports,
                "total_components": self.total_components,
                "total_hooks": self.total_hooks,
                "total_dependencies": len(self.total_dependencies),
                "total_imports": self.total_imports,
            },
            "function_params": self.function_params,
            "function_returns": self.function_returns,
            "class_methods": self.class_methods,
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({"graph": data, "metadata": metadata}, f, indent=2)

    def visualize_graph(self):
        """Visualize the knowledge graph."""
        try:
            import matplotlib.pyplot as plt

            # Create color map for different node types
            color_map = {
                "file": "#ADD8E6",       # Light blue
                "class": "#90EE90",      # Light green
                "function": "#FFE5B4",   # Peach
                "export": "#FFB6C1",     # Light pink
                "component": "#E6E6FA",  # Lavender
                "hook": "#DDA0DD",       # Plum
                "entity": "#FFD700",     # Gold
                "dependency_file": "#C0C0C0",  # Silver
                "dependency": "#8A2BE2",  # Blue Violet
            }

            # Set node colors
            node_colors = [
                color_map.get(self.graph.nodes[node].get("type", "file"), "lightgray")
                for node in self.graph.nodes()
            ]

            # Create figure and axis explicitly
            fig, ax = plt.subplots(figsize=(20, 15))

            # Calculate layout
            pos = nx.spring_layout(self.graph, k=1.5, iterations=50)

            # Draw the graph
            nx.draw(
                self.graph,
                pos,
                ax=ax,
                with_labels=True,
                node_color=node_colors,
                node_size=2000,
                font_size=8,
                font_weight="bold",
                arrows=True,
                edge_color="gray",
                arrowsize=20,
            )

            # Add legend
            legend_elements = [
                plt.Line2D(
                    [0], [0],
                    marker='o',
                    color='w',
                    markerfacecolor=color,
                    label=node_type,
                    markersize=10
                )
                for node_type, color in color_map.items()
            ]

            # Place legend outside the plot
            ax.legend(
                handles=legend_elements,
                loc='center left',
                bbox_to_anchor=(1.05, 0.5),
                title="Node Types"
            )

            # Set title
            ax.set_title("Code Knowledge Graph Visualization", pad=20)

            # Adjust layout to accommodate legend
            plt.subplots_adjust(right=0.85)

            # Show plot
            plt.show()

        except ImportError:
            print("Matplotlib is required for visualization. Install it using 'pip install matplotlib'.")

if __name__ == "__main__":
    try:
        # Directory containing the JavaScript/TypeScript codebase.
        print("Code Knowledge Graph Generator")
        print("-----------------------------")
        codebase_dir = input("Enter the path to the codebase directory: ").strip()

        if not os.path.exists(codebase_dir):
            raise ValueError(f"Directory does not exist: {codebase_dir}")

        output_file = "code_knowledge_graph.json"

        # Create and analyze the codebase.
        print("\nAnalyzing codebase...")
        ckg = JSCodeKnowledgeGraph(directory=codebase_dir)
        ckg.analyze_codebase()

        # Save in standard format.
        print("\nSaving graph...")
        ckg.save_graph(output_file)
        print(f"\nCode knowledge graph saved to {output_file}")

        # Display metadata stats
        print("\nCodebase Statistics:")
        print("-------------------")
        stats = {
            "Total Files": ckg.total_files,
            "Total Classes": ckg.total_classes,
            "Total Functions": ckg.total_functions,
            "Total Hooks": ckg.total_hooks,
            "Total Exports": ckg.total_exports,
            "Total Imports": ckg.total_imports,
            "Total Dependencies": len(ckg.total_dependencies),
        }

        # Calculate max length for padding
        max_len = max(len(key) for key in stats.keys())

        # Print stats in aligned columns
        for key, value in stats.items():
            print(f"{key:<{max_len + 2}}: {value:,}")

        # Optional visualization.
        while True:
            visualize = input("\nWould you like to visualize the graph? (yes/no): ").strip().lower()
            if visualize in ["yes", "no"]:
                break
            print("Invalid choice. Please enter yes or no.")

        if visualize == "yes":
            print("\nGenerating visualization...")
            ckg.visualize_graph()

    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
    except Exception as e:
        print(f"\nError: {str(e)}", file=sys.stderr)
    finally:
        print("\nDone.")

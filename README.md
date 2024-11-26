# 🧠 CntxtJS: Minify Your Codebase Context for LLMs

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=flat-square)](http://makeapullrequest.com)

> 🤯 **75% Token Reduction In Context Window Usage!** Supercharge your LLM's understanding of JavaScript/TypeScript codebases. CntxtJS generates comprehensive knowledge graphs that help LLMs navigate and comprehend your code structure with ease. 

It's like handing your LLM the cliff notes instead of a novel.

## ✨ Features

- 🔍 Deep analysis of JavaScript/TypeScript codebases
- 📊 Generates detailed knowledge graphs of:
  - File relationships and dependencies
  - Class hierarchies and methods
  - Function signatures and parameters
  - React components and hooks
  - Import/export relationships
  - Package dependencies
- 🎯 Specially designed for LLM context windows
- 📈 Built-in visualization capabilities of your projects knowledge graph
- 🚀 Support for a large number of modern JS frameworks and patterns

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/brandondocusen/CntxtJS.git

# Navigate to the directory
cd CntxtJS

# Install required packages
pip install networkx matplotlib

# Run the analyzer
python CntxtJS.py
```

When prompted, enter the path to your JavaScript/TypeScript codebase. The tool will generate a `code_knowledge_graph.json` file and offer to visualize the relationships.

## 💡 Example Usage with LLMs

The LLM can now provide detailed insights about your codebase's implementations, understanding the relationships between components, functions, and files!
After generating your knowledge graph, you can upload it as a single file to give LLMs deep context about your codebase. Here's a powerful example prompt:

```Prompt Example
Based on the knowledge graph, explain how the authentication flow works in this application, 
including which components and functions are involved in the process.
```

```Prompt Example
Based on the knowledge graph, map out the core user experience flow - starting from the landing page through to the core-experience components and their interactions.
```

```Prompt Example
Using the knowledge graph, analyze the state management approach in this application. Which stores exist, what do they manage, and how do they interact with components?
```

```Prompt Example
From the knowledge graph data, break down this application's UI component hierarchy, focusing on reusable elements and their implementation patterns.
```

```Prompt Example
According to the knowledge graph, identify all error handling patterns in this codebase - where are errors caught, how are they processed, and how are they displayed to users?
```

```Prompt Example
Based on the knowledge graph's dependency analysis, outline the key third-party libraries this project relies on and their primary use cases in the application.
```

```Prompt Example
Using the knowledge graph's function analysis, explain how the application handles data fetching and caching patterns across different components.
```

## 📊 Output Format

The tool generates two main outputs:
1. A JSON knowledge graph (`js_code_knowledge_graph.json`)
2. Optional visualization using matplotlib

The knowledge graph includes:
- Detailed metadata about your codebase
- Node and edge relationships
- Function parameters and return types
- Component hierarchies
- Import/export mappings

## 🤝 Contributing

We love contributions! Whether it's:
- 🐛 Bug fixes
- ✨ New features
- 📚 Documentation improvements
- 🎨 Visualization enhancements

Just fork, make your changes, and submit a PR. Check out our [contribution guidelines](CONTRIBUTING.md) for more details.

## 🎯 Future Goals

- [ ] Deeper support for additional frameworks (Vue, Svelte)
- [ ] Enhanced TypeScript type analysis
- [ ] Interactive web-based visualizations
- [ ] Custom graph export formats
- [ ] Integration with popular IDEs

## 📝 License

MIT License - feel free to use this in your own projects!

## 🌟 Show Your Support

If you find CntxtJS helpful, give it a star! ⭐️ 

---

Made with ❤️ for the LLM and JavaScript communities

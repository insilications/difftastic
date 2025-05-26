# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Difftastic is a structural diff tool that compares files based on their syntax rather than just textual differences. It parses code into Abstract Syntax Trees (ASTs) using Tree-sitter parsers and performs intelligent diffing that understands programming language semantics.

## Common Development Commands

### Build and Test
```bash
# Build the project
cargo build

# Build optimized release version
cargo build --release

# Run all tests
cargo test

# Run CLI integration tests specifically
cargo test --test cli

# Run sample file comparison tests
./sample_files/compare_all.sh
```

### Development Tasks (via justfile)
```bash
# List available commands
just

# Serve documentation locally
just doc

# Run comparison tests
just compare

# Generate man page
just man

# Serve homepage locally
just home
```

### Running Difftastic
```bash
# Basic file comparison
./target/debug/difft file1.js file2.js

# Different display modes
./target/debug/difft --display=inline file1.js file2.js
./target/debug/difft --display=side-by-side file1.js file2.js

# JSON output formats
./target/debug/difft --display=json file1.js file2.js

# Check for syntactic changes only (fast)
./target/debug/difft --check-only --exit-code file1.js file2.js
```

## Core Architecture

### Module Organization

**Main Entry Point**: `src/main.rs` - CLI parsing and orchestration

**Core Modules**:
- `parse/` - Language detection, AST manipulation, Tree-sitter integration
- `diff/` - Core diffing algorithms (Dijkstra's algorithm in `dijkstra.rs`)  
- `display/` - Output formatting (side-by-side, inline, JSON)
- `lsp/` - Language Server Protocol implementation

**Key Files**:
- `src/diff/dijkstra.rs` - Core structural diffing algorithm
- `src/parse/tree_sitter_parser.rs` - Tree-sitter parser integration
- `src/parse/guess_language.rs` - Language detection from file extensions
- `src/display/side_by_side.rs` - Default two-column output format

### Language Support

Languages are supported via Tree-sitter parsers in `vendored_parsers/`. The build system (`build.rs`) compiles 29 vendored parsers from C/C++ source during compilation. Language detection happens in `src/parse/guess_language.rs` based on file extensions and content.

### Diffing Algorithm

Difftastic treats structural diffing as a graph problem using Dijkstra's algorithm (`src/diff/dijkstra.rs`). It builds a graph of possible edits and finds the minimum-cost path representing the optimal diff.

## Testing Strategy

### Test Structure
- **Unit tests**: Embedded in source files with `#[cfg(test)]`
- **CLI tests**: `tests/cli.rs` using `assert_cmd` crate
- **Sample comparisons**: `sample_files/` directory with paired test files

### Sample Files
The `sample_files/` directory contains extensive test cases:
- Paired files: `*_1.ext` and `*_2.ext` for each language
- `compare_all.sh` runs comprehensive comparison tests
- Expected outputs in `compare.expected`

### Environment Variables for Testing
- `DFT_LOG` - Logging configuration
- `DFT_PARSE_ERROR_LIMIT` - Allow parse errors during testing
- `DFT_BACKGROUND` - Terminal color scheme (light/dark)

## Build System

### Custom Build Process
The `build.rs` script:
- Compiles 29 Tree-sitter parsers from C/C++ source
- Uses parallel compilation via `rayon`
- Handles mixed C/C++ compilation requirements
- Embeds git commit information

### Cross-Compilation
- Supports cross-compilation via `CARGO_TARGET_*_RUNNER` 
- CLI tests detect and use cross-compilation runners
- Windows-specific compiler flags handled in build script

## Performance Considerations

### Memory Usage
- Uses `MiMalloc` allocator for better performance
- Memory-intensive on large files due to graph-based algorithm
- Configurable limits via environment variables:
  - `DFT_GRAPH_LIMIT` - Maximum graph size
  - `DFT_BYTE_LIMIT` - Maximum file size for structural diffing

### Optimization Features
- `--check-only` mode for fast syntactic change detection
- Slider algorithms in `src/diff/sliders.rs` for improved output
- Unchanged region detection in `src/diff/unchanged.rs`

## Development Workflow

1. **Language Support**: Add new Tree-sitter parsers to `vendored_parsers/` and update `build.rs`
2. **Display Changes**: Modify files in `src/display/` for output formatting
3. **Algorithm Improvements**: Work in `src/diff/` modules, especially `dijkstra.rs`
4. **Testing**: Add sample files to `sample_files/` and run `compare_all.sh`

## Special Features

### Merge Conflict Resolution
Difftastic can parse and diff merge conflict markers (`<<<<<<<`, `=======`, `>>>>>>>`):
```bash
difft file_with_conflicts.js
```

### LSP Integration
The `src/lsp/` module provides Language Server Protocol support for editor integration with capabilities like real-time diffing.

### Fallback Behavior
When Tree-sitter parsing fails, difftastic falls back to line-oriented text diffing with word highlighting to ensure reliable operation.
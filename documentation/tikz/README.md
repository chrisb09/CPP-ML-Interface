# Thesis Figure Drafts

The construction/runtime figure is now split into independently includable
PDFs for [the coupling guide](../coupling_guide.md) and potential thesis use:

| Source | Content | Approximate PDF size |
| --- | --- | --- |
| `coupling_construction.tex` | Horizontal ownership tree and side legend | 150 x 54 mm |
| `coupling_flow.tex` | Typed static step, solver fields and control loop | 150 x 119 mm |

Both use Latin Modern and the architecture figure's palette/connector sizes.
Class text is 9 pt, operations/notes 7 pt, and edge annotations 6 pt. Each
is authored at its final 150 mm inclusion width, not downscaled from the
former combined canvas. Build the construction figure with:

```sh
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build coupling_construction.tex
cp build/coupling_construction.pdf coupling_construction.pdf
pdftoppm -png -singlefile -r 150 coupling_construction.pdf coupling_construction
```

Build the runtime figure with:

```sh
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build coupling_flow.tex
cp build/coupling_flow.pdf coupling_flow.pdf
pdftoppm -png -singlefile -r 150 coupling_flow.pdf coupling_flow
```

This schematic describes ordinary static orchestration, not flexible staging
or the specialized FlowExtrapolator history/forecast algorithm. Its arrows
show logical data movement, not mandatory copies or automatic conversion.
The construction figure is a horizontal ownership tree with its own scalar-type
abbreviation key and an in-graph composition-symbol legend.
The runtime figure distinguishes rectangular class containers from rounded green
function boxes and branches on the boolean returned by
`Behavior::should_perform_inference()`, not a separate decision object.
The figures can be placed separately in the thesis. Operations are shown
within Application and Library rather than as separate classes.
The runtime entry is `MLCoupling::step()` delegating to `Application::ml_step()`.
Library and Solver share the bottom row, with Library, Application and
MLCoupling left-aligned. Solver contains orange `coupling_input: Data<CI>` and
`coupling_output: Data<CO>` fields: LI/LO are Application's model-side buffers,
not the solver's raw fields. The solver input routes under Library to the
preparation hook; its output receives the finalized data through an orthogonal connector. A dashed
solver control loop returns to `step()` for the next coupling step. The grey
`Continue simulation` operation represents solver activity, not a CMI method.
Connectors use
orthogonal routing with explicit bends rather than diagonal approximations.
Preparation/finalization wrappers, their hooks, and `Behavior::time_step_delta()`
are included. Pure virtual functions (`should_perform_inference()`,
`time_step_delta()`, and `static_inference()`) use red italic text. Virtual
functions with a base implementation, such as `ml_step()` and the
preprocessing/postprocessing hooks, use black italic text. Non-virtual
functions (`step()` and the preparation/finalization wrappers) remain black
and upright. Italics distinguish virtual dispatch; red additionally marks
pure virtual interfaces.
Preserve the 150 mm physical width when evaluating the 9/7/6 pt font hierarchy.
For example, use `\includegraphics[width=150mm]{coupling_flow.pdf}`. Figure
captions belong in the thesis floats; these standalone graphics do not
contain embedded subcaptions.

`simplified_architecture.tex` is a standalone TikZ figure. Its fixed layout
follows the schematic reference, with UML ownership and inheritance relationships
corrected to match the implementation. The existing PlantUML diagrams are separate.

From this directory, build the vector PDF and PNG preview with:

```sh
mkdir -p build
pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build simplified_architecture.tex
cp build/simplified_architecture.pdf simplified_architecture.pdf
pdftoppm -png -singlefile -r 150 simplified_architecture.pdf simplified_architecture
```

Requires LaTeX with `standalone`, TikZ and Latin Modern, plus Poppler's
`pdftoppm`. Intermediate files remain in the ignored `build/` directory.
The PDF can be included directly with `\includegraphics` in a thesis.

## Thesis Typography

The figure follows `~/Master-Thesis/notes/thesis-figures-guide.md`: Latin Modern
with T1 encoding and embedded Type 1 fonts, authored at approximately 150 mm
wide rather than relying on downscaling a large canvas. Latin Modern Sans
matches existing architecture/topology figures in the thesis. This dense
diagram uses 9 pt class text/headings, 8 pt relationship-symbol legend text,
7 pt explanatory notes, and 6 pt edge labels, a consistent
local reduction from the guide's default 11/10/9 pt hierarchy. Main class names
wrap after `MLCoupling` to preserve legibility at the final inclusion width.
Use `\includegraphics[width=150mm]{simplified_architecture.pdf}`; inspect the
result at printed size. Colors retain the UML abstract/concrete distinction.

The application base is concrete. Library, behavior and normalization bases
are abstract. `MLCoupling` owns the application, library and behavior; the
application owns optional normalization and four primary data containers.
The flow-extrapolator application also has working storage that is omitted here.
Data containers contain tensor objects; the diagram does not specify whether
their backing memory is owned or borrowed. The C API wraps the C++
coupling class, and the Fortran module wraps the C API.

Dashed links from Application to Library and Behavior show the borrowed
interfaces used by `ml_step()` for inference and scheduling. Crossing lines
without a junction do not indicate a relationship. Library and Normalization
also use Data in their interfaces; these additional container dependencies
are omitted to keep the overview focused on the component relationships.

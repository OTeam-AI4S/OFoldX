# Working rule

Think before you code. Now is better than never, although never is better than
right now.

Every delegated task prompt repeats that exact sentence verbatim. Pointing an
agent to this file is not a substitute for putting the principle in its task.

Read the relevant parts of `DESIGN.md` and think before editing code.

Work in explicit Tik–Tok cycles. Tik establishes the simplest complete bounded
contract justified by the evidence currently needed for that implementation.
Tok implements that accepted contract and lets practice test its ergonomics,
ownership, performance, and integration. Do not start Tok before thinking, but do
not extend Tik in pursuit of proof that no future evidence could change the design.

In `DESIGN.md`, every normative subsection is investigation by default. An
implementation task must name an explicit `Status: frozen` subsection and its
current evidence; unmarked prose cannot authorize code. `frozen` means accepted as
the implementation basis for this Tik–Tok cycle, not eternal truth. New contrary
evidence or implementation friction reopens the subsection before the affected Tok
continues.

Every delegated task must say explicitly which of these it is:

- **READ-ONLY INVESTIGATION — DO NOT EDIT FILES.** State the bounded question,
  authoritative evidence, and required findings.
- **IMPLEMENTATION.** State the frozen contract and revisions, exact file ownership,
  exclusions, and commands or evidence required for acceptance.

When evidence invalidates an implementation premise, stop the affected edit and
return to Tik. Implementation, parity, and tests are legitimate evidence about a
design; they must not be used as substitutes for thinking or as excuses to preserve
a premise they have falsified.

`DESIGN.md` is a working design investigation with known falsified sections, not
an accepted contract or evidence that its prose is correct. Investigation and
review prompts must try to falsify the relevant design
against authoritative model behavior and ordinary user intuition, not merely check
that code conforms to it. Report a design defect before proposing an implementation
workaround.

Previously rejected designs are regression cases, not discarded history. Review
the invariant and ownership pattern rather than its current spelling: renaming a
request wrapper, metadata layer, capability taxonomy, generic validation pass, or
lifecycle abstraction does not justify reintroducing it. A review that checks only
new lines or names is incomplete.

Record every high-impact finding that changes a public contract, invalidates an
earlier premise, or materially changes a family-completion route in `REVIEW.md` so
the project owner can review it without rereading the full investigation. That file
is a short review surface, not a second specification: accepted normative wording
still belongs in `DESIGN.md`, and confirmed upstream defects belong in
`UPSTREAM_BUGS.md`.

This is a research library: trust normal Python and PyTorch contracts and let
ordinary failures fail. Validate only scientific ambiguity or invariants that normal
operations cannot express; do not add defensive scans, wrappers, fallbacks, or
lifecycle machinery without that evidence. Performance is a first-class constraint.
An explicit device, dtype, or operation argument is experienced-user intent. Obey
it directly: do not warn, reject, repair, preserve hidden precision exceptions, or
infer a supposedly safer choice. Mixed precision belongs to AMP or the caller's
explicit module choices.

For Model, Auto/loading/saving, and Runner patterns, compare MultiMolecule first and
start from deletion. Every OFoldX-only abstraction must have a concrete
structure-domain invariant or an existing consumer that cannot be expressed more
directly.

A family audit starts by independently inventorying released checkpoints, command
paths, and scientific functions from an exact inspected upstream revision. That
revision makes the audit evidence reproducible; it is not an OFoldX dependency or
public-behavior pin. The audit then computes
the difference against OFoldX. Local code and family READMEs are objects under
review; they do not define or narrow the authoritative upstream surface.

Deletion of a generic framework, Condition, helper, or shared vocabulary never
authorizes deletion of the official function it carried badly. Move that function
to its exact family Processor, Model/Module, Result, concrete Pipeline, converter,
or repository surface and retain it completely. KISS removes accidental machinery,
not scientific capability.

The inventory is the union of released command paths, public Model/Module methods,
checkpoint variants, conditioning modes, heads, and decoded outputs. A shipped
configuration or CLI default does not erase a public scientific parameter merely
because that path currently fixes it to one value. Reviewers must trace the
underlying callable and actively search for a counterexample to every proposed axis,
default, and ownership boundary before accepting its removal.

Private files and helpers, dataclasses, hidden state, `compare=False`, and custom
`__post_init__` mutation each carry an explicit burden of proof. Prefer public or
inline ordinary Python and ordinary value semantics. Code files do not define
`__all__`.

Review in two passes: first scientific correctness and complete support against the
frozen evidence; then subtraction, intuition, and defensiveness. The second pass
removes every noun, branch, guard, and layer that lacks a necessary consumer or
invariant.

# OLM: Open-Loop Memory for Reliable Long-Horizon Language Agents

Long-horizon language agents increasingly rely on memory systems to store,
retrieve, and update past experience across sessions. However, existing
memory architectures primarily treat memory as a source of relevant
information, and do not explicitly preserve the unresolved conditions under
which past conclusions were produced. As a result, agents may later use
unverified assumptions, insufficient evidence, unresolved contradictions,
deferred commitments, scope-limited solutions, or partially verified results
as if they were settled facts. We propose **OLM**
(**O**pen-**L**oop **M**emory), a memory control layer that
treats unresolved epistemic obligations as first-class memory objects. OLM
maintains open loops for unresolved conditions and activates them when a
future claim, action, or memory use requires their closure. Activated loops
can constrain reasoning, require targeted verification, block unsafe actions,
or restrict the evidentiary use of otherwise valid semantic memories until
the relevant condition is closed. OLM implements this lifecycle through
three operators, `Open`, `RequiresClosed`, and `Close`,
together with typed gates and memory-license invalidation over the closed
knowledge store. Experiments on obligation-aware overlays of public
long-horizon memory and agent benchmarks show that OLM reduces obligation
violations and improves safe action blocking over strong memory and
verification-gated baselines, while remaining competitive on standard
long-term memory tasks.

## Contributions

- We identify *evidence licensing* as a distinct failure mode in
  long-horizon agent memory. The key insight is that agents may
  retrieve factually correct memories but reuse them as stronger evidence
  than the unresolved verification state permits.

- We propose OLM, a lifecycle-managed memory control layer based on
  open loops. OLM represents unresolved epistemic obligations as
  persistent memory objects and regulates their effect through
  `Open`, `RequiresClosed`, and `Close`, combining
  obligation-conditioned activation, typed gates, and memory license
  invalidation.

- We show that OLM improves both standard long-term memory
  performance and obligation-aware memory use. Across controlled
  evaluations, OLM reduces obligation violations, improves hard-block
  recall, and remains competitive on ordinary memory tasks compared with
  strong retrieval-oriented and verification-gated baselines.

## OLM

![OLM Framework](resource/OLM.pdf)

## Methodology

OLM augments a conventional closed store $\mathcal{K}_t$ with an open-loop
store $\mathcal{L}_t$. Closed memories are retrieved as evidence; open
loops activate when the current decision would depend on, contradict, or
be invalidated by an unresolved condition. A memory in $\mathcal{K}_t$
encodes what may be used; a loop in $\mathcal{L}_t$ encodes
what is not yet entitled to be assumed.

OLM is organized around three interfaces: `Open`,
`RequiresClosed`, and `Close`. We instantiate all three as structured
LLM classifiers; a lightweight embedding prefilter is applied before
invoking `RequiresClosed` to keep pairwise queries tractable.

### Memory as a Constrained Decision Process

At each step $t$, the standard memory-augmented policy is:

$$
a_t \sim \pi_0(a \mid x_t, R_{\mathcal{K}}(x_t))
$$

OLM first assigns each candidate action and draft claim a license state.
Actions whose required loops remain unresolved are removed from the
admissible set unless a low-cost verification action is available; the
base policy then selects only among admissible actions:

$$
a_t \sim \pi_{\text{OLM}}
(a \mid x_t, R_{\mathcal{K}}(x_t), A_{\mathcal{L}}(x_t)),
\quad
a_t \in \Omega_t
$$

where $A_{\mathcal{L}}(x_t)\subseteq \mathcal{L}_t$ is the activated loop
set and $\Omega_t$ is the admissible action set after loop constraints.

### Open-Loop Representation

An open loop is a persistent record of an unresolved obligation:

$$
\ell =
(id_\ell,\,r_\ell,\,p_\ell,\,E_\ell,\,T_\ell,\,q_\ell,\,
 g_\ell,\,\rho_\ell,\,s_\ell,\,\Gamma_\ell,\,\mathcal{M}_\ell,\,\pi_\ell)
$$

The loop type $r_\ell$ (one of six values: `assumption`,
`evidence_gap`, `contradiction`,
`deferred_commitment`, `scope_limit`,
`partial_verification`) governs evidence aggregation,
unresolvedness scoring, and closure semantics.
The proposition $p_\ell$ is the unresolved claim; $E_\ell$ the signed
evidence set; $T_\ell$ the typed trigger schema; $q_\ell$ the closure
predicate; $g_\ell$ the gate constraint; $\rho_\ell$ a risk score;
$\Gamma_\ell$ scope conditions; $\mathcal{M}_\ell$ the closed-memory ids
whose licensed use is conditioned on this loop; and $\pi_\ell$ provenance.

The closure predicate $q_\ell$ must be *operational*---evaluable
from future evidence under a declared modality (`tool-verifiable`,
`external-query-verifiable`, `evidence-threshold`, or
`monitorable`). It returns:

$$
q_\ell(E_\ell) \in
\{\texttt{confirmed},\,\texttt{refuted},\,\texttt{superseded},\,\texttt{insufficient}\}
$$

with the first three outcomes closing the loop. The lifecycle state
$$
s_\ell \in \{\texttt{open},\,\texttt{investigating},\,
\texttt{closed\_confirmed},\,\texttt{closed\_refuted},\,
\texttt{superseded},\,\texttt{stale},\,\texttt{expired}\}
$$
prevents unbounded accumulation via `stale` and `expired`
transitions.

The admissibility condition $\text{Adm}(\ell)$ requires: (i)
$\mathcal{O}(q_\ell)$---the closure predicate is evaluable; (ii)
$T_\ell \neq \emptyset$---at least one trigger condition exists; (iii)
$\rho_\ell > \theta_\rho \vee \text{HighStake}(T_\ell)$---the
obligation is consequential or its triggers reference irreversible
actions; and (iv) $\Delta(\ell,\mathcal{L}_t) > \theta_{\text{new}}$---
the loop is sufficiently distinct from existing ones.

### Opening and Activating Loops

`Open` extracts candidate obligations from a trajectory segment:

$$
\widehat{\mathcal{L}}_{i:j} = \text{Open}(\tau_{i:j}, \mathcal{K}_t)
$$

Loop creation occurs post-session and before high-stakes actions.

Activation is governed by `RequiresClosed`. OLM collects decision
objects:

$$
\mathcal{D}_t = \mathcal{D}^{\text{act}}_t \cup
\mathcal{D}^{\text{claim}}_t \cup \mathcal{D}^{\text{mem}}_t
$$

For each $d \in \mathcal{D}_t$ and loop $\ell$, `RequiresClosed`
returns:

$$
(b_{\ell,d},\, \nu_{\ell,d},\, c_{\ell,d},\, \mathcal{E}^*_{\ell,d})
$$

A loop activates when $b_{\ell,d} = 1$ and
$c_{\ell,d} > \theta_{\text{act}}$ for some $d$. Under context budget
pressure, activated loops are prioritized by

$$
\omega_\ell(t)
=
\rho_\ell \cdot u_\ell \cdot
\max_{d \in \mathcal{D}_t} c_{\ell,d} \cdot
\exp[-\lambda_{\text{age}}(t - t_\ell)]
$$

where $u_\ell \in [0,1]$ is a type-specific unresolvedness score.

### Loop-Gated Reasoning and Memory Use

The relation type $\nu_{\ell,d}$ routes each activated loop to one of
three gate types:

| $\nu_{\ell,d}$ | Gate type | Effect |
|---|---|---|
| `claim_dep` | Epistemic | Hedge, qualify, or verify before claiming |
| `act_precond` | Action | Block or route through hard gate |
| `mem_license` | License | Downgrade $\kappa_i \in \mathcal{M}_\ell$ license status |
| $\mathcal{E}^*_{\ell,d}\neq\emptyset$, low cost | Inquiry | Issue targeted verification |

For irreversible actions, a two-phase hard gate applies:

$$
\text{Gate}(a, \ell) =
\begin{cases}
  \texttt{allow}, & b_{\ell,a} = 0, \\
  \texttt{verify}(v^\star), & \mathcal{E}^*_{\ell,a} \neq \emptyset
    \;\wedge\; \text{cost}(v^\star) \leq \delta, \\
  \texttt{hedge\_or\_defer}, & a \text{ is reversible or linguistic}, \\
  \texttt{block}, & \text{otherwise.}
\end{cases}
$$

If the result is $\texttt{verify}(v^\star)$, the runtime executes $v^\star$,
updates loop evidence via `Close`, and re-evaluates the gate.

Open loops regulate closed memory by downgrading *license status*:

$$
\text{License}(\kappa_i, d, \mathcal{A}_t) \in
\{\texttt{usable},\, \texttt{context\_only},\, \texttt{requires\_qualification},\, \texttt{blocked}\}
$$

assigned when $\kappa_i \in \mathcal{M}_\ell$ and $\nu_{\ell,d} =
`mem_license`. This separates factual content from evidentiary
role: ``unit test passed after parser patch'' remains factually correct,
but an activated `partial_verification` loop prevents it from
licensing ``root cause was resolved.''

### Closing Loops and Consolidating Memory

`Close` appends signed evidence and updates loop state. When a
loop closes, OLM distills a resolution record and revises affected closed
memories via one of four updates ordered from least to most destructive:
`preserve`, `downgrade_license`, `attach_scope`,
or `tombstone`. Separating `downgrade_license` from
confidence reduction ensures that memories whose facts are correct but
whose evidentiary role is invalidated are not erroneously weakened.

### Interface Specification

`Open` maps a trajectory segment and closed store to typed
candidate loops, each carrying an operational closure predicate and at
least one trigger entry. `RequiresClosed` is the central predicate:

$$
\text{RequiresClosed}(d, \ell) \to (b, \nu, c, \mathcal{E}^*)
$$

where $b \in \{0,1\}$ is the closure-requirement decision; $\nu$ routes
to the appropriate gate; $c \in [0,1]$ is confidence; and $\mathcal{E}^*$
is sufficient evidence for closure. `Close` maps new observations
to updated evidence and a closure verdict.

These contracts constitute the algorithmic contribution of OLM and admit
symbolic checkers, static analyzers, or learned classifiers as alternative
instantiations. The TSM+VerifyGate baseline receives the same verification
backbone and uses per-entry scope tags and hedging at retrieval time, but
lacks obligation lifecycle management and obligation-conditioned
activation---isolating OLM's specific contribution.

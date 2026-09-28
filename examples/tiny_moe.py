"""CPU data-flow demo: 4 tokens, 4 experts, top-2, 2 simulated ranks.

Run with Python 3.9+ and PyTorch 2.x: python examples/tiny_moe.py
No distributed runtime, accelerator, or Mega-MoE-TD installation is used.
"""

import torch
import torch.nn.functional as functional


WORLD = 2
EXPERTS = 4
TOPK = 2
TOKENS_PER_RANK = 2


def naive_moe(x, weights, w1, w2, selected):
    """Independent reference: each token calls its two logical experts.

    W1[e] is [H, 2F], W2[e] is [H, F]. Weight the expert OUTPUT here.
    """
    outputs = []
    for token in range(x.shape[0]):
        contributions = []
        for slot in range(TOPK):
            expert = int(selected[token, slot])
            gate, up = (x[token] @ w1[expert]).chunk(2, dim=-1)
            expert_out = (functional.silu(gate) * up) @ w2[expert].T
            contributions.append(weights[token, slot] * expert_out)
        outputs.append(torch.stack(contributions).sum(dim=0))
    return torch.stack(outputs)


def make_plan(selected, destinations):
    """Stable (destination, expert, source, route_id) order; Python metadata."""
    routes = range(selected.numel())
    order = sorted(
        routes,
        key=lambda route: (
            int(destinations.flatten()[route]),
            int(selected.flatten()[route]),
            (route // TOPK) // TOKENS_PER_RANK,
            route,
        ),
    )
    counts = [[0] * WORLD for _ in range(WORLD)]
    for route in routes:
        source = (route // TOPK) // TOKENS_PER_RANK
        counts[source][int(destinations.flatten()[route])] += 1
    return order, counts


def bucketed_moe(x, weights, w1, w2, selected, destinations):
    """Simulate dispatch, per-physical-bucket GEMMs, and reverse combine.

    Weight the ACTIVATION here. A replica clone stays connected to its
    logical owner's autograd graph, so PyTorch sums its weight gradients.
    This models the reduction result, not real replica transport.
    """
    order, _ = make_plan(selected, destinations)
    flat_experts = selected.flatten()
    flat_destinations = destinations.flatten()
    values, route_ids = [], []
    for rank in range(WORLD):
        for expert in range(EXPERTS):
            ids = [route for route in order
                   if int(flat_destinations[route]) == rank
                   and int(flat_experts[route]) == expert]
            if not ids:
                continue
            ids = torch.tensor(ids, dtype=torch.long)
            recv_x = x[ids // TOPK]
            local_w1, local_w2 = w1[expert], w2[expert]
            if rank != expert // (EXPERTS // WORLD):
                local_w1, local_w2 = local_w1.clone(), local_w2.clone()
            gate, up = (recv_x @ local_w1).chunk(2, dim=-1)
            weighted_act = functional.silu(gate) * up * weights.flatten()[ids, None]
            values.append(weighted_act @ local_w2.T)
            route_ids.append(ids)

    # Restore route_id order first, then sum slots; do NOT weight twice.
    returned = torch.zeros(x.shape[0] * TOPK, x.shape[1], dtype=x.dtype)
    returned = returned.index_copy(0, torch.cat(route_ids), torch.cat(values))
    return returned.reshape(x.shape[0], TOPK, x.shape[1]).sum(dim=1)


def print_plan(name, selected, destinations):
    order, counts = make_plan(selected, destinations)
    loads = [sum(row[rank] for row in counts) for rank in range(WORLD)]
    print(f"\n{name}: order={order}; send_counts={counts}; rank_loads={loads}")
    print("dst  offset  expert  token  slot  route_id  src")
    offsets = [0] * WORLD
    for route in order:
        token, slot = divmod(route, TOPK)
        rank = int(destinations[token, slot])
        expert = int(selected[token, slot])
        print(rank, offsets[rank], expert, token, slot, route,
              token // TOKENS_PER_RANK, sep="      ")
        offsets[rank] += 1


def check_forward_and_gradients(name, actual, reference, parameters):
    """Compare outputs and vector-Jacobian products for a nonuniform dy."""
    torch.testing.assert_close(actual, reference, rtol=1e-10, atol=1e-12)
    dy = torch.linspace(-0.7, 0.9, actual.numel(), dtype=actual.dtype).reshape_as(actual)
    actual_grads = torch.autograd.grad((actual * dy).sum(), parameters)
    reference_grads = torch.autograd.grad((reference * dy).sum(), parameters)
    for label, grad, expected in zip(("x", "route_weights", "W1", "W2"),
                                     actual_grads, reference_grads):
        torch.testing.assert_close(grad, expected, rtol=1e-10, atol=1e-12,
                                   msg=f"{name}: {label}")
    print(f"PASS {name}: output, dx, d_route_weights, dW1, dW2")


def check_selective_recompute(x, weights, w1, selected, destinations):
    """Rebuild expanded inputs and weighted activations without a second FC1."""
    order, _ = make_plan(selected, destinations)
    ids = torch.tensor(order, dtype=torch.long)
    experts = selected.flatten()[ids]
    with torch.no_grad():
        # Forward: FC1 is evaluated once. These two expanded tensors are
        # retained ONLY as verification oracles in this teaching script.
        recv_x = x[ids // TOPK]
        saved_y = torch.stack([recv_x[row] @ w1[int(expert)]
                               for row, expert in enumerate(experts)])
        gate, up = saved_y.chunk(2, dim=-1)
        act = functional.silu(gate) * up * weights.flatten()[ids, None]

        # The production idea: keep original x, route mapping/weights, Y;
        # reconstruct recv_x and act when backward needs them.
        saved = {"x": x.clone(), "route_ids": ids.clone(),
                 "weights": weights.flatten()[ids].clone(), "fc1_output": saved_y.clone()}
        rebuilt_x = saved["x"][saved["route_ids"] // TOPK]
        gate, up = saved["fc1_output"].chunk(2, dim=-1)
        rebuilt_act = functional.silu(gate) * up * saved["weights"][:, None]
        torch.testing.assert_close(rebuilt_x, recv_x, rtol=0, atol=0)
        torch.testing.assert_close(rebuilt_act, act, rtol=0, atol=0)
    print("PASS selective recompute: recv_x [8,2], act [8,2]; FC1 was not rerun")


def main():
    # CPU + float64 isolate algebra/layout differences from BF16 rounding.
    generator = torch.Generator().manual_seed(7)
    x = torch.tensor([[1.0, 0.5], [-0.5, 1.0], [0.2, -0.8], [0.7, 0.3]],
                     dtype=torch.float64, requires_grad=True)
    weights = torch.tensor([[0.75, 0.25], [0.60, 0.40], [0.50, 0.50], [0.20, 0.80]],
                           dtype=torch.float64, requires_grad=True)
    w1 = (torch.randn(EXPERTS, 2, 4, generator=generator, dtype=torch.float64) * 0.3
          ).requires_grad_()
    w2 = (torch.randn(EXPERTS, 2, 2, generator=generator, dtype=torch.float64) * 0.3
          ).requires_grad_()
    parameters = (x, weights, w1, w2)
    balanced = torch.tensor([[0, 2], [1, 2], [0, 3], [1, 3]])
    skewed = torch.tensor([[0, 1], [0, 2], [0, 1], [0, 3]])

    for name, selected, use_replica in (
        ("balanced", balanced, False),
        ("skewed / owner only", skewed, False),
        ("skewed / e0 replica on rank 1", skewed, True),
    ):
        destinations = selected // (EXPERTS // WORLD)
        if use_replica:
            destinations = destinations.clone()
            destinations[2:, 0] = 1  # Move routes 4 and 6; logical expert stays e0.
        print_plan(name, selected, destinations)
        actual = bucketed_moe(*parameters, selected, destinations)
        reference = naive_moe(*parameters, selected)
        print("output:", actual.detach().tolist())
        check_forward_and_gradients(name, actual, reference, parameters)

    check_selective_recompute(x, weights, w1, balanced, balanced // 2)


if __name__ == "__main__":
    main()

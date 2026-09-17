import torch

from event_rl.event_smdp import event_smdp_value_loss


def test_smdp_value_target_discounts_duration_and_stops_bootstrap_at_terminal():
    values = torch.tensor([1.0, 1.0, 4.0, 9.0], requires_grad=True)
    rewards = torch.tensor([1.0, 2.0, 3.0])
    dones = torch.tensor([False, False, False, True])
    events = torch.tensor([0, 0, 1])
    loss, target, starts = event_smdp_value_loss(values, rewards, dones, events, gamma=.9)
    torch.testing.assert_close(target, torch.tensor([1.0 + .9 * 2.0 + .9**2 * 4.0, 3.0]))
    torch.testing.assert_close(starts, torch.tensor([0, 2]))
    loss.backward()
    assert values.grad[0] != 0 and values.grad[2] != 0
    assert values.grad[1] == 0 and values.grad[3] == 0

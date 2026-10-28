"""Slice plot labels: instantaneous, averaged and fluctuating pressure share the name 'pr'."""

from chapsim2_toolkit.slice import variable_colorbar_label, variable_label


def test_pressure_labels_say_which_pressure():
    assert (variable_label('pr'), variable_colorbar_label('pr')) == ('Pressure', '$p$')
    assert (variable_label("pr'"), variable_colorbar_label("pr'")) == ('Pressure Fluctuation', "$p'$")
    assert variable_label('t_avg_pr') == variable_label('tsp_avg_pr') == 'Time-averaged Pressure'
    assert variable_colorbar_label('t_avg_pr') == r'$\langle p \rangle$'

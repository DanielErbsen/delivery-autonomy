from autonomy.controller import CONTROLLER_PARAMS, controller_param_problem


def test_defaults_are_accepted():
    assert controller_param_problem(**CONTROLLER_PARAMS) is None


def test_speeds_must_stay_inside_mppi_bounds():
    for name, value in [('max_speed', 0.0), ('max_speed', -0.3), ('max_speed', 1.6),
                        ('min_speed', 0.1), ('min_speed', -0.6)]:
        assert controller_param_problem(**{**CONTROLLER_PARAMS, name: value}) is not None, (name, value)


def test_slow_and_forward_only_settings_are_accepted():
    assert controller_param_problem(**{**CONTROLLER_PARAMS, 'max_speed': 0.8, 'min_speed': 0.0}) is None


def test_tolerances_must_be_positive():
    for name in ('xy_goal_tolerance', 'yaw_goal_tolerance'):
        assert controller_param_problem(**{**CONTROLLER_PARAMS, name: 0.0}) is not None, name

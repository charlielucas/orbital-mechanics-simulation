from streamlit.testing.v1 import AppTest


def test_default_app_renders_without_exceptions() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()

    assert not app.exception
    assert app.title[0].value == "Orbital Mechanics Explorer"
    assert app.sidebar.selectbox[0].value == "circular_reference"
    assert any("Both measured drifts stay below" in message.value for message in app.success)
    assert any(
        button.label == "Download complete trajectory" for button in app.get("download_button")
    )


def test_j2_guided_scenario_renders_rate_comparison() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()

    app.sidebar.selectbox[0].select("j2_precession").run()

    assert not app.exception
    assert app.sidebar.selectbox[0].value == "j2_precession"
    assert any("fitted nodal rate is within 2%" in message.value for message in app.success)
    assert any(metric.label == "Fitted RAAN rate" for metric in app.metric)


def test_custom_scenario_uses_exploratory_language() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()

    app.sidebar.selectbox[0].select("custom").run()

    assert not app.exception
    assert app.sidebar.radio[0].value == "Two-body"
    assert any("not a checked-in acceptance case" in message.value for message in app.info)


def test_custom_polar_j2_scenario_explains_near_zero_rate() -> None:
    app = AppTest.from_file("app.py", default_timeout=30).run()

    app.sidebar.selectbox[0].select("custom").run()
    app.sidebar.radio[0].set_value("Two-body plus J2")
    app.sidebar.slider[2].set_value(90.0)
    app.run()

    assert not app.exception
    assert any(metric.label == "Fitted RAAN rate" for metric in app.metric)
    assert any("effectively zero" in message.value for message in app.info)
    assert not any("undefined for this equatorial orbit" in message.value for message in app.info)

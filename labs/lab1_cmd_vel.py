"""Lab 1 — velocity / cmd_vel interface + camera-follow controller.

The robotics substrate: the discrete symbolic manoeuvres are translated here into
a continuous velocity setpoint ``(v, ω)`` that maps directly onto ROS
``geometry_msgs/Twist`` (``linear.x = v``, ``angular.z = ω``) — the interface a
real robot is driven through. This is the bridge that lets all of Part 2's
symbolic logic survive the move to MetaDrive / a robot in Part 3.

Function-only. The ROS publishing side and the camera-follow perception are
stubbed (no ROS in Colab); the manoeuvre→velocity translation is fully usable.
"""


def manoeuvre_to_cmd_vel(manoeuvre, scene, cfg):
    """Translate a discrete meta-action into a velocity setpoint ``(v, ω)``.

    Keeps the FSM/shield reasoning over manoeuvres (Part 2) while actuating
    through velocity (Part 3 / a robot). Speeds/turn-rates come from the YAML.

    Args:
        manoeuvre: one of ``LANE_LEFT, IDLE, LANE_RIGHT, FASTER, SLOWER``.
        scene: scene dict (uses ego current speed as the baseline).
        cfg: full config.

    Returns:
        ``(v, ω)`` in SI units (m/s, rad/s).
    """
    md = cfg["metadrive"]
    v_max = md["v_max"]
    omega_max = md["omega_max"]
    v_cur = scene["ego"].get("v", 0.0)
    v_cruise = 0.6 * v_max
    v_base = max(v_cur, v_cruise)

    dv = 0.2 * v_max  # speed step per FASTER/SLOWER
    table = {
        "IDLE": (v_base, 0.0),
        "FASTER": (v_base + dv, 0.0),
        "SLOWER": (v_base - dv, 0.0),
        "LANE_LEFT": (v_base, +omega_max),
        "LANE_RIGHT": (v_base, -omega_max),
    }
    v, omega = table.get(manoeuvre, (v_cur, 0.0))
    v = float(max(0.0, min(v, v_max)))
    omega = float(max(-omega_max, min(omega, omega_max)))
    return v, omega


def to_twist(v, omega):
    """Return a ``geometry_msgs/Twist``-shaped dict (ROS-agnostic).

    ``linear.x = v``, ``angular.z = ω``; all other components zero. A ROS node
    would publish this on ``cmd_vel``.
    """
    return {
        "linear": {"x": float(v), "y": 0.0, "z": 0.0},
        "angular": {"x": 0.0, "y": 0.0, "z": float(omega)},
    }


def publish_cmd_vel(twist, topic="cmd_vel"):
    """Publish a Twist to a ROS topic.

    TODO: wire to ``rospy``/``rclpy`` (e.g. a ``geometry_msgs/Twist`` publisher on
    ``cmd_vel``) using the Lab 1 stack. Stubbed — there is no ROS in Colab.
    """
    raise NotImplementedError("TODO: ROS cmd_vel publisher (Lab 1 stack, not in Colab).")


def camera_follow_controller(image, cfg):
    """Minimal 'follow a target' controller to sanity-check the perception loop.

    TODO: implement the Lab 1 camera-follow (detect target in ``image`` → produce
    ``(v, ω)`` to keep it centred). Stubbed — the Lab 1 vision API was unavailable.
    """
    raise NotImplementedError("TODO: Lab 1 camera-follow controller (handout API unknown).")

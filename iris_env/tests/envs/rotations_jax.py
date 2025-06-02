import jax
import jax.numpy as jnp

"""
This JAX adaption of rotations.py has the following changes/improvements: 
1. All NumPy functions have been replaced with their JAX equivalents 
2. In-place operations, which are not supported on JAX, have been replaced with funcitonal
updates using .at[].set() or .at[].multiply()
3. Note that random number generation now usews JAX's random number generator which requries
a PRNG key
4. The quaternion_slerp function has been vectorized using jax.vmap to allow for batch processing
5. Control flows have been rewritten to use JAX's functional style and array operations
6. All functions are pure functions (requirement for automatic diff and JIT compilation)

Note that these functions will only work with JAX arrays and not Numpy Arrays
Note that random number generation will require a PRNG Key
"""

# Converter to quaternion from (radian angle, direction)
@jax.jit
def angle_dir_to_quat(angle, dir):
    dir = dir / jnp.linalg.norm(dir)
    quat = jnp.zeros(4)
    quat = quat.at[0].set(jnp.cos(angle / 2))
    quat = quat.at[1:].set(jnp.sin(angle / 2) * dir)
    return quat

@jax.jit
def rpy_to_quaternion(angles):
    yaw, pitch, roll = angles[0], angles[1], angles[2]

    qx = jnp.sin(roll/2) * jnp.cos(pitch/2) * jnp.cos(yaw/2) - jnp.cos(roll/2) * jnp.sin(pitch/2) * jnp.sin(yaw/2)
    qy = jnp.cos(roll/2) * jnp.sin(pitch/2) * jnp.cos(yaw/2) + jnp.sin(roll/2) * jnp.cos(pitch/2) * jnp.sin(yaw/2)
    qz = jnp.cos(roll/2) * jnp.cos(pitch/2) * jnp.sin(yaw/2) - jnp.sin(roll/2) * jnp.sin(pitch/2) * jnp.cos(yaw/2)
    qw = jnp.cos(roll/2) * jnp.cos(pitch/2) * jnp.cos(yaw/2) + jnp.sin(roll/2) * jnp.sin(pitch/2) * jnp.sin(yaw/2)

    return jnp.array([qw, qx, qy, qz])

@jax.jit
def quat_to_rpy(q):
    x, y, z, w = q[1], q[2], q[3], q[0]
    roll = jnp.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = jnp.arcsin(jnp.clip(2 * (w * y - x * z), -1, 1))
    yaw = jnp.arctan2(2 * (w * z + x * y), 1 - 2 * (z * z + y * y))
    return jnp.array([roll, pitch, yaw])

@jax.jit
def axisangle2quat(axisangle):
    dir = axisangle[0:3]
    angle = axisangle[3]
    dir = dir / jnp.linalg.norm(dir)
    quat = jnp.zeros(4)
    quat = quat.at[0].set(jnp.cos(angle / 2))
    quat = quat.at[1:].set(jnp.sin(angle / 2) * dir)
    return quat

# Quaternion multiplication
@jax.jit
def quaternion_multiply(quaternion1, quaternion0):
    w0, x0, y0, z0 = quaternion0
    w1, x1, y1, z1 = quaternion1
    return jnp.array([
        -x1 * x0 - y1 * y0 - z1 * z0 + w1 * w0,
        x1 * w0 + y1 * z0 - z1 * y0 + w1 * x0,
        -x1 * z0 + y1 * w0 + z1 * x0 + w1 * y0,
        x1 * y0 - y1 * x0 + z1 * w0 + w1 * z0])

@jax.jit
def quaternion_mat(q):
    return jnp.array([
        [q[0], -q[1], -q[2], -q[3]],
        [q[1],  q[0], -q[3],  q[2]],
        [q[2],  q[3],  q[0], -q[1]],
        [q[3], -q[2],  q[1],  q[0]],
    ])

@jax.jit
def quaternionToAxisAngle(p):
    p = p / jnp.linalg.norm(p)
    angle = 2 * jnp.arccos(p[0])
    axis = p[1:] / jnp.sin(angle / 2)
    return axis, angle

@jax.jit
def quaternion_mul(q1, q2):
    return quaternion_mat(q1) @ q2

@jax.jit
def quaternion_conjugate(quaternion):
    return quaternion.at[1:].multiply(-1)

@jax.jit
def quaternion_inverse(quaternion):
    return quaternion_conjugate(quaternion) / jnp.dot(quaternion, quaternion)

@jax.jit
def quaternion_real(quaternion):
    return quaternion[0]

@jax.jit
def quaternion_imag(quaternion):
    return quaternion[1:4]

@jax.jit
def quaternion_slerp(quat0, quat1, frac):
    def slerp(frac, q0, q1):
        cos_half_theta = jnp.dot(q0, q1)
        q1 = jnp.where(cos_half_theta < 0, -q1, q1)
        cos_half_theta = jnp.abs(cos_half_theta)
        
        half_theta = jnp.arccos(cos_half_theta)
        sin_half_theta = jnp.sqrt(1.0 - cos_half_theta * cos_half_theta)
        
        ratio_a = jnp.sin((1 - frac) * half_theta) / sin_half_theta
        ratio_b = jnp.sin(frac * half_theta) / sin_half_theta
        
        return ratio_a * q0 + ratio_b * q1

    return jax.vmap(slerp, in_axes=(0, None, None))(frac, quat0, quat1)

@jax.jit
def random_quaternion(key):
    rand = jax.random.uniform(key, (3,))
    r1 = jnp.sqrt(1.0 - rand[0])
    r2 = jnp.sqrt(rand[0])
    pi2 = jnp.pi * 2.0
    t1 = pi2 * rand[1]
    t2 = pi2 * rand[2]
    return jnp.array([jnp.cos(t2) * r2, jnp.sin(t1) * r1,
                      jnp.cos(t1) * r1, jnp.sin(t2) * r2])

@jax.jit
def quat2rotmat(Q):
    q0, q1, q2, q3 = Q[0], Q[1], Q[2], Q[3]

    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    rot_matrix = jnp.array([[r00, r01, r02],
                            [r10, r11, r12],
                            [r20, r21, r22]])

    return rot_matrix

@jax.jit
def quat2angle(q):
    return 2.0 * jnp.arccos(q[0]) * jnp.sign(q[-1])

@jax.jit
def angle2mat(angle):
    return jnp.array([
        [jnp.cos(angle), -jnp.sin(angle)],
        [jnp.sin(angle),  jnp.cos(angle)]
    ])

# Conjugate quaternion matrix
@jax.jit
def conjquat_mat_fn(q):
    return jnp.array([
        [ q[0], -q[1], -q[2], -q[3]],
        [ q[1],  q[0],  q[3], -q[2]],
        [ q[2], -q[3],  q[0],  q[1]],
        [ q[3],  q[2], -q[1],  q[0]]
    ])

@jax.jit
def conjquatmat_wb_fn(wb):
    return jnp.array([
        [    0, -wb[0], -wb[1], -wb[2]],
        [wb[0],      0,  wb[2], -wb[1]],
        [wb[1], -wb[2],      0,  wb[0]],
        [wb[2],  wb[1], -wb[0],     0]
    ])

# Quaternion to DCM
@jax.jit
def quat2dcm_fn(q):
    return jnp.array([
        [1 - 2*(q[2]**2 + q[3]**2), 2*(q[1]*q[2] - q[0]*q[3]), 2*(q[1]*q[3] + q[0]*q[2])],
        [2*(q[1]*q[2] + q[0]*q[3]), 1 - 2*(q[1]**2 + q[3]**2), 2*(q[2]*q[3] - q[0]*q[1])],
        [2*(q[1]*q[3] - q[0]*q[2]), 2*(q[2]*q[3] + q[0]*q[1]), 1 - 2*(q[1]**2 + q[2]**2)]
    ])

# Position to homogeneous transformation matrix
@jax.jit
def ttmat_fn(pos):
    return jnp.array([
        [1, 0, 0, pos[0]],
        [0, 1, 0, pos[1]],
        [0, 0, 1, pos[2]],
        [0, 0, 0, 1]
    ])

# Rotations to homogeneous transformation matrices
@jax.jit
def rxtmat_fn(alpha):
    return jnp.array([
        [1, 0, 0, 0],
        [0, jnp.cos(alpha), -jnp.sin(alpha), 0],
        [0, jnp.sin(alpha),  jnp.cos(alpha), 0],
        [0, 0, 0, 1]
    ])

@jax.jit
def rytmat_fn(beta):
    return jnp.array([
        [ jnp.cos(beta), 0, jnp.sin(beta), 0],
        [0, 1, 0, 0],
        [-jnp.sin(beta), 0, jnp.cos(beta), 0],
        [0, 0, 0, 1]
    ])

@jax.jit
def rztmat_fn(theta):
    return jnp.array([
        [jnp.cos(theta), -jnp.sin(theta), 0, 0],
        [jnp.sin(theta),  jnp.cos(theta), 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Quaternion to homogeneous transformation matrix
@jax.jit
def quattmat_fn(q):
    dcm = quat2dcm_fn(q)
    return jnp.block([
        [dcm, jnp.zeros((3, 1))],
        [jnp.zeros((1, 3)), jnp.ones((1, 1))]
    ])
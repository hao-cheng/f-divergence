"""This script contains helper loss functions for QA."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf
import modeling


def margin_func(t, lb, ub):
    """Computes min(max(lb, t), ub)."""

    def hs_func(x, s):
        """Computes max(0, s - x)."""
        return tf.nn.relu(s - x)

    return hs_func(t, ub) - hs_func(t, lb)


def compute_margin_loss(logits, one_hot_positions, tau=24, epsilon=1e-5):
    """This computes the margin loss for the correctly classifiy case."""
    label_logits = tf.reduce_sum(one_hot_positions * logits, axis=-1)

    best_wrong_logits = tf.reduce_logsumexp(
        tf.log(1.0 - one_hot_positions) + logits, axis=-1
    )

    mask = tf.to_float(label_logits > best_wrong_logits)
    margin_loss = (
        tf.reduce_sum(mask * margin_func(label_logits - best_wrong_logits, 0, tau))
        + epsilon
    )
    num_correct = tf.reduce_sum(mask) + epsilon
    margin_loss /= num_correct
    return margin_loss, mask


def compute_pos_margin_loss(logits, one_hot_positions, tau=24, epsilon=1e-5):
    """This computes the margin loss for the correctly classifiy case."""
    label_logits = one_hot_positions * logits + (one_hot_positions - 1.0) * logits
    return -tf.reduce_sum(margin_func(label_logits, epsilon, tau))


def l2_normalizer(v, epsilon1=1e-6, epsilon2=1e-12):
    """Normalizes the vector into L2 norm ball."""
    v /= (tf.reduce_max(tf.abs(v), axis=-1, keepdims=True) + epsilon1)
    v /= (tf.sqrt(tf.reduce_sum(tf.square(v), axis=-1, keepdims=True))
          + epsilon2)
    return v


def get_random_norm_vector(shape, epsilon1=1e-6, epsilon2=1e-12):
    v = tf.random_normal(shape=shape)
    return l2_normalizer(v)


def compute_flatten_jacobian(y, x, v):
    """Computes Jy/x using random vector v."""
    flat_target = tf.reshape(y * v, [-1])
    flat_jacobian = tf.gradients(flat_target, [x])[0]

    return flat_jacobian


def compute_jacobian_loss_w_multiple_iter(
    logits, inputs, num_class, n_iter=1, is_training=False, tape=None
):
    """Computes jacobian Frobenius norm."""
    output_shape = logits.shape.as_list()
    num_class = tf.to_float(num_class)

    if is_training:
        random_vec_list = [get_random_norm_vector(output_shape) for _ in range(n_iter)]
    else:
        random_vec_list = tf.unstack(
            tf.one_hot(tf.range(num_class), depth=num_class, dtype=tf.float32), axis=0
        )
        n_iter = num_class

    jacobian_loss_list = []
    for random_vec in random_vec_list:
        # jacobian = compute_flatten_jacobian(logits, inputs, random_vec)
        jacobian = tape.gradient(tf.reshape(logits * random_vec, [-1]), [inputs])[0]
        jacobian_loss_list.append(
            tf.reduce_mean(
                num_class
                * tf.reduce_sum(tf.square(jacobian), axis=tf.range(1, tf.rank(inputs)))
                / n_iter
            )
        )

    jacobian_loss = tf.reduce_sum(tf.stack(jacobian_loss_list))
    return jacobian_loss


def kl_divergence(p, log_p, log_q, axis=-1):
    """Computes KL divergence. BP through Q."""
    p = tf.stop_gradient(p)
    log_p = tf.stop_gradient(log_p)
    return tf.reduce_sum(p * (log_p - log_q), axis=axis)


def rev_kl_divergence(p, log_p, log_q, axis=-1):
    """Computes reverse-KL divergence. BP through P."""
    log_q = tf.stop_gradient(log_q)
    # log_p = tf.stop_gradient(log_p)
    return tf.reduce_sum(p * (log_p - log_q), axis=axis)


def kl_divergence_w_logits(
    logits_p, logits_q, label_weights=None, axis=-1, epsilon=1e-6,
    reverse_kl=False,
):
    """Computes KL divergence."""
    logits_p = tf.cast(logits_p, dtype=tf.float32)
    logits_q = tf.cast(logits_q, dtype=tf.float32)

    log_p = tf.nn.log_softmax(logits_p, axis=axis)
    log_q = tf.nn.log_softmax(logits_q, axis=axis)

    if reverse_kl:
        q = tf.nn.softmax(logits_q, axis=-1)
        kl_div = rev_kl_divergence(q, log_q, log_p, axis=axis)
    else:
        p = tf.nn.softmax(logits_p, axis=-1)
        kl_div = kl_divergence(p, log_p, log_q, axis=axis)

    if label_weights is not None:
        label_weights = tf.reshape(label_weights, [-1])
        num = tf.reduce_sum(kl_div * label_weights)
        dem = tf.reduce_sum(label_weights) + epsilon
        kl_div = num / dem
    else:
        kl_div = tf.reduce_mean(kl_div)

    return kl_div


def symmetric_kl(logits_p, logits_q, label_weights=None, axis=-1, epsilon=1e-6):
    """Computes symmetric KL."""
    logits_p = tf.cast(logits_p, dtype=tf.float32)
    logits_q = tf.cast(logits_q, dtype=tf.float32)

    p = tf.nn.softmax(logits_p, axis=-1)
    log_p = tf.nn.log_softmax(logits_p, axis=-1)
    q = tf.nn.softmax(logits_q, axis=-1)
    log_q = tf.nn.log_softmax(logits_q, axis=-1)

    per_example_loss = 0.5 * (
        kl_divergence(p, log_p, log_q, axis=axis)
        + kl_divergence(q, log_q, log_p, axis=axis)
        # + rev_kl_divergence(q, log_q, log_p, axis=axis)
    )

    if label_weights is not None:
        label_weights = tf.reshape(label_weights, [-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + epsilon
        loss = numerator / denominator
    else:
        loss = tf.reduce_mean(per_example_loss)

    return loss


def l2_normalize_vector(d, epsilon1=1e-6, epsilon2=1e-12, reduce_batch=False):
    """Normalizes vector."""
    if reduce_batch:
        reduce_axis = tf.range(1, tf.rank(d))
    else:
        reduce_axis = -1
    d /= (epsilon1 + tf.reduce_max(tf.abs(d), axis=reduce_axis, keep_dims=True))
    d /= tf.sqrt(epsilon2 + tf.reduce_sum(tf.square(d), axis=reduce_axis, keep_dims=True))
    return d


def l1_normalize_vector(d, epsilon=1e-6, reduce_batch=False):
    """Normalizes vector in L1."""
    if reduce_batch:
        reduce_axis = tf.range(1, tf.rank(d))
    else:
        reduce_axis = -1

    d /= (epsilon + tf.reduce_sum(tf.abs(d), axis=reduce_axis, keep_dims=True))
    return d


def linf_normalize_vector(d, epsilon=1e-6, reduce_batch=False):
    """Normalizes vector in Linf."""
    if reduce_batch:
        reduce_axis = tf.range(1, tf.rank(d))
    else:
        reduce_axis = -1

    d /= (epsilon + tf.reduce_max(tf.abs(d), axis=reduce_axis, keep_dims=True))
    return d


def normalize_vector(d, normalizer="L2", reduce_batch=False):
    """Normalizes the vector into norm ball."""
    if normalizer == "L2":
      return l2_normalize_vector(d, reduce_batch=reduce_batch)
    elif normalizer == "L1":
      return l1_normalize_vector(d, reduce_batch=reduce_batch)
    elif normalizer == "Linf":
      return linf_normalize_vector(d, reduce_batch=reduce_batch)
    else:
        raise ValueError("Unknown normalizer %s" % normalizer)


def generate_noise(x, normalizer="L2", reduce_batch=False):
    d = tf.random_normal(shape=tf.shape(x))
    return normalize_vector(d, normalizer=normalizer, reduce_batch=reduce_batch)


def get_embeddings(model, noise_normalizer, noise_epsilon, noise_type="tok"):
    """Gets the model embeddings with or without random noise."""
    if noise_epsilon > 0:
        if noise_type == "tok":
            rand_noise = tf.stop_gradient(
                noise_epsilon * generate_noise(
                    model.get_embedding_output(), normalizer=noise_normalizer,
                    reduce_batch=False))
        elif noise_type == "seq":
            rand_noise = tf.stop_gradient(
                noise_epsilon * generate_noise(
                    model.get_embedding_output(), normalizer=noise_normalizer,
                    reduce_batch=True)
            )
        elif noise_type == "tok+seq":
            seq_noise = tf.stop_gradient(
                tf.sqrt(noise_epsilon) * generate_noise(
                    model.get_embedding_output(), normalizer=noise_normalizer,
                    reduce_batch=True))
            tok_noise = tf.stop_gradient(
                noise_epsilon * generate_noise(
                    model.get_embedding_output(), normalizer=noise_normalizer,
                    reduce_batch=False))
            rand_noise = seq_noise + tok_noise
        else:
            raise ValueError("Unknown noise type %s" % noise_type)

        embeddings = model.get_embedding_output() + rand_noise
    else:
        embeddings = model.get_embedding_output()

    return tf.stop_gradient(embeddings)


def compute_forward_logits(pooled_output, num_labels, output_var_scope="cls",
                           is_training=True, output_layer_dropout=0.9,
                           reuse=True):
    """Computes the logits."""
    final_hidden_shape = pooled_output.shape.as_list()
    batch_size = final_hidden_shape[0]
    hidden_size = final_hidden_shape[1]

    with tf.variable_scope(output_var_scope, reuse=reuse):
        output_weights = tf.get_variable(
            "output_weights", [num_labels, hidden_size],
            initializer=tf.truncated_normal_initializer(stddev=0.02))

        output_bias = tf.get_variable(
            "output_bias", [num_labels], initializer=tf.zeros_initializer())

        if is_training:
            tf.logging.info("Using output_layer_keep_prob %f" %
                            output_layer_dropout)
            pooled_output = tf.nn.dropout(
                pooled_output, keep_prob=output_layer_dropout)

        logits = tf.matmul(pooled_output, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        logits = tf.cast(logits, dtype=tf.float32)

    return logits


def compute_jacobian_loss(model, num_labels, output_var_scope="cls"):
    """Computes Jacobian loss."""
    with tf.GradientTape(persistent=False) as tape:
        embedding_inputs = model.get_embedding_output()
        tape.watch(embedding_inputs)
        _, pooled_output = model.adv_forward(embedding_inputs)

        logits = compute_forward_logits(pooled_output, num_labels)

    jacobian = tape.batch_jacobian(logits, [embedding_inputs])[0]
    jacobian_loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(jacobian), axis=tf.range(1, tf.rank(jacobian))))

    return jacobian_loss


def compute_approx_jacobian_norm(logits, inputs):
    """Computes approximate Frobenius norm of Jacobian."""
    random_vector = generate_noise(logits)
    final_hidden_shape = logits.shape.as_list()
    batch_size = final_hidden_shape[0]
    num_labels = final_hidden_shape[1]

    flat_logits = tf.reshape(logits, [-1])
    flat_rand = tf.reshape(random_vector, [-1])

    sampled_jacobian = tf.gradients(flat_logits * flat_rand, [inputs])[0]

    jacobian_norm = tf.reduce_sum(
        tf.square(sampled_jacobian)) / tf.to_float(batch_size)

    return jacobian_norm


def compute_weighted_jacobian_norm(logits, inputs):
    """Computes approximate Frobenius norm of Jacobian."""
    random_vector = generate_noise(logits)
    final_hidden_shape = logits.shape.as_list()
    batch_size = final_hidden_shape[0]
    num_labels = final_hidden_shape[1]

    log_prob = tf.nn.log_softmax(logits)
    weight = tf.stop_gradient(tf.exp(-0.5 * log_prob))
    flat_logits = tf.reshape(logits * weight , [-1])
    flat_rand = tf.reshape(random_vector, [-1])

    sampled_jacobian = tf.gradients(flat_logits * flat_rand, [inputs])[0]

    jacobian_norm = tf.reduce_sum(
        tf.square(sampled_jacobian)) / tf.to_float(batch_size)

    return jacobian_norm


def contrastive_project(x, output_dim, is_training=False, keep_prob=1.0,
                        initializer_range=0.02, act_func="relu",
                        scope="contrastive_projection_layer", reuse=None):
    """Computes a contrastive project of input."""
    with tf.variable_scope(scope or "contrastive_layer", reuse=reuse):
        if is_training:
            tf.logging.info("Using contrastive_layer_keep_prob %f" %
                            keep_prob)
            x = tf.nn.dropout(x, keep_prob=keep_prob)

        y = tf.layers.dense(
            x,
            output_dim,
            kernel_initializer=modeling.create_initializer(initializer_range),
            activation=modeling.get_activation(act_func),
            name="non_linear_dense",
        )

        if is_training:
            y = tf.nn.dropout(y, keep_prob=keep_prob)

        y = tf.layers.dense(
            y,
            output_dim,
            kernel_initializer=modeling.create_initializer(initializer_range),
            activation=None,
            use_bias=False,
            name="linear_dense",
        )

        return y


def compute_vat_loss(logits, num_labels, model,
                     is_training=True,
                     output_layer_dropout=0.9,
                     output_var_scope="cls",
                     loss_type="v3",
                     noise_normalizer="L2",
                     rand_noise_epsilon=1e-3,
                     noise_epsilon=1e-3):
    """Computes the double forward loss."""
    rand_noise = rand_noise_epsilon * generate_noise(
            model.get_embedding_output(), normalizer="L2")
    embeddings = model.get_embedding_output() + rand_noise

    _, pooled_output = model.adv_forward(embeddings)

    d_logits = compute_forward_logits(
        pooled_output, num_labels, output_var_scope=output_var_scope,
        is_training=is_training, output_layer_dropout=output_layer_dropout,
        reuse=True,
    )

    loss_func = kl_divergence_w_logits
    if loss_type == "js":
        tf.logging.info("Using js loss for VAT")
        loss_func = js_divergence
    elif loss_type == "hellinger":
        tf.logging.info("Using hellinger loss for VAT")
        loss_func = hellinger_distance
    else:
        tf.logging.info("Using VAT with KL-divergence")

    perturb_loss = loss_func(logits, d_logits)

    perturb = tf.gradients(perturb_loss, [rand_noise])[0]
    perturb = tf.stop_gradient(
        normalize_vector(perturb, normalizer=noise_normalizer))

    embeddings = model.get_embedding_output() + noise_epsilon * perturb
    _, pooled_output = model.adv_forward(embeddings)
    v_logits = compute_forward_logits(
        pooled_output, num_labels, output_var_scope=output_var_scope,
        is_training=is_training, output_layer_dropout=output_layer_dropout,
        reuse=True,
    )

    vat_loss = loss_func(logits, v_logits)

    return vat_loss


def compute_double_forward_loss_v1_w_add_noise(
        logits, num_labels, model,
        noise_normalizer="L2", noise_epsilon=1e-5, output_var_scope="cls"):
    """Computes the double forward loss."""
    _, pooled_output = model.adv_forward(
        get_embeddings(model, noise_normalizer, noise_epsilon))

    d_logits = compute_forward_logits(pooled_output, num_labels)

    double_forward_loss = symmetric_kl(logits, d_logits)

    return double_forward_loss


def compute_double_forward_loss_v2_w_add_noise(logits, num_labels, model,
                                               output_var_scope="cls",
                                               noise_normalizer="L2",
                                               noise_epsilon=1e-5):
    """Computes the double forward loss."""
    _, pooled_output = model.adv_forward(
        get_embeddings(model, noise_normalizer, noise_epsilon))

    d_logits = compute_forward_logits(pooled_output, num_labels)

    double_forward_loss = kl_divergence_w_logits(logits, d_logits, reverse_kl=True)

    return double_forward_loss


def compute_double_forward_loss_v3_w_add_noise(logits, num_labels, model,
                                               output_var_scope="cls",
                                               noise_normalizer="L2",
                                               noise_epsilon=1e-5):
    """Computes the double forward loss."""
    _, pooled_output = model.adv_forward(
        get_embeddings(model, noise_normalizer, noise_epsilon))

    d_logits = compute_forward_logits(pooled_output, num_labels)

    double_forward_loss = kl_divergence_w_logits(logits, d_logits)

    return double_forward_loss


def alpha_beta_kl_divergence_with_logits(
    logits_p, logits_q, alpha=1.0, beta=1.0, label_weights=None, axis=-1,
    epsilon=1e-6,
):
    """Computes alpha-beta KL divergence."""
    logits_p = tf.cast(logits_p, dtype=tf.float32)
    logits_q = tf.cast(logits_q, dtype=tf.float32)

    log_p = tf.nn.log_softmax(logits_p, axis=axis)
    log_q = tf.nn.log_softmax(logits_q, axis=axis)

    p = tf.nn.softmax(logits_p, axis=axis)

    kl_div = tf.reduce_sum(
        alpha * p * tf.stop_gradient(log_p) -
        beta * tf.stop_gradient(p) * log_q, axis=axis)

    if label_weights is not None:
        label_weights = tf.reshape(label_weights, [-1])
        num = tf.reduce_sum(kl_div * label_weights)
        dem = tf.reduce_sum(label_weights) + epsilon
        kl_div = num / dem
    else:
        kl_div = tf.reduce_mean(kl_div)

    return kl_div


def hellinger_distance(logits, d_logits):
    # clean_prob = tf.nn.softmax(0.5 * logits)
    # noise_prob = tf.nn.softmax(0.5 * d_logits)
    # loss = tf.reduce_mean(
    #     0.5 * tf.reduce_sum(tf.square(clean_prob -
    #                                   noise_prob), axis=-1)
    # )

    clean_prob = tf.nn.softmax(logits)
    noise_prob = tf.nn.softmax(d_logits)
    product_term = tf.sqrt(clean_prob * noise_prob)
    sum_term = 0.5 * (clean_prob + noise_prob)

    loss = tf.reduce_mean(
        tf.reduce_sum(sum_term - product_term, axis=-1))
    return loss


def js_divergence(logits, d_logits):
    clean_prob = tf.nn.softmax(logits)
    clean_log_prob = tf.nn.log_softmax(logits)
    noise_prob = tf.nn.softmax(d_logits)
    noise_log_prob = tf.nn.log_softmax(d_logits)
    mean_prob = 0.5 * (clean_prob + noise_prob)
    mean_log_prob = tf.stop_gradient(tf.log(mean_prob))
    loss = tf.reduce_mean(
        0.5 * kl_divergence(clean_prob, clean_log_prob,
                            mean_log_prob) +
        0.5 * kl_divergence(noise_prob, noise_log_prob,
                            mean_log_prob)
    )
    return loss


def compute_double_forward_loss_w_add_noise(
        logits, num_labels, model, loss_type="v3",
        noise_type="seq",
        alpha=1.0, beta=1.0, output_layer_dropout=0.9, is_training=True,
        noise_normalizer="L2", noise_epsilon=1e-5, output_var_scope="cls"):
    """Computes the double forward loss."""
    _, pooled_output = model.adv_forward(
        get_embeddings(model, noise_normalizer, noise_epsilon,
                       noise_type=noise_type))

    d_logits = compute_forward_logits(
        pooled_output, num_labels, output_var_scope=output_var_scope,
        is_training=is_training, output_layer_dropout=output_layer_dropout,
        reuse=True,
    )

    if loss_type == "v1":
        tf.logging.info("Using double forward loss v1")
        double_forward_loss = symmetric_kl(logits, d_logits)
    elif loss_type == "v2":
        tf.logging.info("Using double forward loss v2")
        double_forward_loss = kl_divergence_w_logits(logits, d_logits, reverse_kl=True)
    elif loss_type == "v3":
        tf.logging.info("Using double forward loss v3")
        double_forward_loss = kl_divergence_w_logits(logits, d_logits)
    elif loss_type == "alpha_beta":
        tf.logging.info("Using alpha-beta KL divergence")
        double_forward_loss = alpha_beta_kl_divergence_with_logits(
            logits, d_logits, alpha=alpha, beta=beta)
    elif loss_type == "hellinger":
        tf.logging.info("Using double forward loss with squared hellinger loss")
        clean_prob = tf.nn.softmax(logits)
        noise_prob = tf.nn.softmax(d_logits)
        double_forward_loss = tf.reduce_mean(
            0.5 * tf.reduce_sum(tf.square(tf.sqrt(clean_prob) -
                                          tf.sqrt(noise_prob)), axis=-1)
        )
    elif loss_type == "js":
        tf.logging.info("Using double forward with Jensen-Shannon divergence")
        # clean_span_prob = tf.stop_gradient(clean_span_prob)
        clean_prob = tf.nn.softmax(logits)
        clean_log_prob = tf.nn.log_softmax(logits)
        noise_prob = tf.nn.softmax(d_logits)
        noise_log_prob = tf.nn.log_softmax(d_logits)
        mean_prob = 0.5 * (clean_prob + noise_prob)
        mean_log_prob = tf.log(mean_prob)
        double_forward_loss = tf.reduce_mean(
            0.5 * kl_divergence(clean_prob, clean_log_prob,
                                mean_log_prob) +
            0.5 * kl_divergence(noise_prob, noise_log_prob,
                                mean_log_prob)
        )

    else:
        raise ValueError("Unknown loss type %s" % loss_type)

    return double_forward_loss

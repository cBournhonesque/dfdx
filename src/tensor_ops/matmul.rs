use crate::prelude::*;
use matrixmultiply::sgemm;

/// Matrix multiplication.
///
/// # Arguments
/// * `lhs` - a 2d tensor representing a MxN matrix
/// * `rhs` - a 2d tensor representing a NxO matrix
///
/// Returns a 2d tensor representing an MxO matrix.
///
/// # Examples
///
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor2D<3, 2> = Tensor2D::zeros();
/// let y: Tensor2D<2, 4> = Tensor2D::zeros();
/// let result: Tensor2D<3, 4> = matmul(x, &y);
/// ```
pub fn matmul<const M: usize, const N: usize, const O: usize, H: Tape>(
    lhs: Tensor2D<M, N, H>,
    rhs: &Tensor2D<N, O, NoneTape>,
) -> Tensor2D<M, O, H> {
    let mut result = Tensor2D::zeros();
    matmat_mul_into(lhs.data(), rhs.data(), result.mut_data());

    // copy rhs data for use later when computing gradients
    let rhs_data = rhs.data.clone();

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    let (lhs, mut tape) = lhs.split_tape();
    tape.add_backward_op(move |grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &_result);
        matmat_mul_into_yt(result_grad, rhs_data.as_ref(), lhs_grad);

        let (rhs_grad, result_grad) = grads.mut_and_ref(&_rhs, &_result);
        matmat_mul_into_xt(lhs.data(), result_grad, rhs_grad);
    });

    result.put_tape(tape)
}

/// Matrix multiplication with the transpose of `rhs`. Equivalent to `matmul(lhs, transpose(rhs))`.
///
/// # Arguments
/// * `lhs` - a 2d tensor representing a MxN matrix
/// * `rhs_t` - a 2d tensor representing a OxN matrix.
///
/// Returns a 2d tensor representing an MxO matrix.
///
/// # Examples
///
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor2D<3, 2> = Tensor2D::zeros();
/// let y: Tensor2D<4, 2> = Tensor2D::zeros();
/// let result: Tensor2D<3, 4> = matmul_transpose(x, &y);
/// ```
pub fn matmul_transpose<const M: usize, const N: usize, const O: usize, H: Tape>(
    lhs: Tensor2D<M, N, H>,
    rhs_t: &Tensor2D<O, N, NoneTape>,
) -> Tensor2D<M, O, H> {
    let mut result = Tensor2D::zeros();
    matmat_mul_into_yt(lhs.data(), rhs_t.data(), result.mut_data());

    // copy rhs data for use later when computing gradients
    let rhs_data = rhs_t.data.clone();

    let _rhs = rhs_t.phantom();
    let _result = result.phantom();
    let (lhs, mut tape) = lhs.split_tape();
    tape.add_backward_op(move |grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &_result);
        matmat_mul_into(result_grad, rhs_data.as_ref(), lhs_grad);

        let (rhs_t_grad, result_grad) = grads.mut_and_ref(&_rhs, &_result);
        matmat_mul_into_xtzt(lhs.data(), result_grad, rhs_t_grad);
    });

    result.put_tape(tape)
}

/// vector * matrix multiplication.
///
/// # Arguments
/// * `lhs` - a 1d tensor representing a 1xN matrix
/// * `rhs` - a 2d tensor representing a NxO matrix
///
/// Returns a 1d tensor representing an 1xO matrix.
///
/// # Examples
///
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor1D<2> = Tensor1D::zeros();
/// let y: Tensor2D<2, 4> = Tensor2D::zeros();
/// let result: Tensor1D<4> = vecmat_mul(x, &y);
/// ```
pub fn vecmat_mul<const N: usize, const O: usize, H: Tape>(
    lhs: Tensor1D<N, H>,
    rhs: &Tensor2D<N, O, NoneTape>,
) -> Tensor1D<O, H> {
    let mut result = Tensor1D::zeros();
    vecmat_mul_into(lhs.data(), rhs.data(), result.mut_data());

    let rhs_data = rhs.data.clone();

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    let (lhs, mut tape) = lhs.split_tape();
    tape.add_backward_op(move |grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &_result);
        vecmat_mul_into_yt(result_grad, rhs_data.as_ref(), lhs_grad);

        let (rhs_t_grad, result_grad) = grads.mut_and_ref(&_rhs, &_result);
        vecvec_mul_into(lhs.data(), result_grad, rhs_t_grad);
    });

    result.put_tape(tape)
}

/// vector * matrix multiplication where `rhs` is transposed. `y * transpose(rhs)`
///
/// # Arguments
/// * `lhs` - a 1d tensor representing a 1xN matrix
/// * `rhs_t` - a 2d tensor representing a OxN matrix
///
/// Returns a 1d tensor representing an 1xO matrix.
///
/// # Examples
///
/// ```rust
/// # use dfdx::prelude::*;
/// let x: Tensor1D<2> = Tensor1D::zeros();
/// let y: Tensor2D<4, 2> = Tensor2D::zeros();
/// let result: Tensor1D<4> = vecmat_mul_transpose(x, &y);
/// ```
pub fn vecmat_mul_transpose<const N: usize, const O: usize, H: Tape>(
    lhs: Tensor1D<N, H>,
    rhs: &Tensor2D<O, N, NoneTape>,
) -> Tensor1D<O, H> {
    let mut result = Tensor1D::zeros();
    vecmat_mul_into_yt(lhs.data(), rhs.data(), result.mut_data());

    let rhs_data = rhs.data.clone();

    let _rhs = rhs.phantom();
    let _result = result.phantom();
    let (lhs, mut tape) = lhs.split_tape();
    tape.add_backward_op(move |grads| {
        let (lhs_grad, result_grad) = grads.mut_and_ref(&lhs, &_result);
        vecmat_mul_into(result_grad, rhs_data.as_ref(), lhs_grad);

        let (rhs_t_grad, result_grad) = grads.mut_and_ref(&_rhs, &_result);
        vecvec_mul_into(result_grad, lhs.data(), rhs_t_grad);
    });

    result.put_tape(tape)
}

/// matrix multiply `x * y`
fn matmat_mul_into<const M: usize, const N: usize, const O: usize>(
    x: &[[f32; N]; M],
    y: &[[f32; O]; N],
    out: &mut [[f32; O]; M],
) {
    unsafe {
        let a = x.as_ptr() as *const f32;
        let b = y.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            M, N, O, 1.0, a, N as isize, 1, b, O as isize, 1, 1.0, c, O as isize, 1,
        )
    };
}

/// matrix multiply `out = transpose(x) * y + beta * out`
fn matmat_mul_into_xt<const M: usize, const N: usize, const O: usize>(
    x_t: &[[f32; M]; N],
    y: &[[f32; O]; N],
    out: &mut [[f32; O]; M],
) {
    unsafe {
        let a = x_t.as_ptr() as *const f32;
        let b = y.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            M, N, O, 1.0, a, 1, M as isize, b, O as isize, 1, 1.0, c, O as isize, 1,
        )
    };
}

/// matrix multiply `x * transpose(y)`
fn matmat_mul_into_yt<const M: usize, const N: usize, const O: usize>(
    x: &[[f32; N]; M],
    y_t: &[[f32; N]; O],
    out: &mut [[f32; O]; M],
) {
    unsafe {
        let a = x.as_ptr() as *const f32;
        let b = y_t.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            M, N, O, 1.0, a, N as isize, 1, b, 1, N as isize, 1.0, c, O as isize, 1,
        )
    };
}

/// matrix multiply `transpose(out) = transpose(x) * y + beta * transpose(out)`
fn matmat_mul_into_xtzt<const M: usize, const N: usize, const O: usize>(
    x_t: &[[f32; M]; N],
    y: &[[f32; O]; N],
    out_t: &mut [[f32; M]; O],
) {
    unsafe {
        let a = x_t.as_ptr() as *const f32;
        let b = y.as_ptr() as *const f32;
        let c = out_t.as_mut_ptr() as *mut f32;
        sgemm(
            M, N, O, 1.0, a, 1, M as isize, b, O as isize, 1, 1.0, c, 1, M as isize,
        )
    };
}

fn vecmat_mul_into<const N: usize, const O: usize>(
    x: &[f32; N],
    y: &[[f32; O]; N],
    out: &mut [f32; O],
) {
    unsafe {
        let a = x.as_ptr() as *const f32;
        let b = y.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            1, N, O, 1.0, a, N as isize, 1, b, O as isize, 1, 1.0, c, O as isize, 1,
        )
    };
}

fn vecmat_mul_into_yt<const N: usize, const O: usize>(
    x: &[f32; N],
    y_t: &[[f32; N]; O],
    out: &mut [f32; O],
) {
    unsafe {
        let a = x.as_ptr() as *const f32;
        let b = y_t.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            1, N, O, 1.0, a, N as isize, 1, b, 1, N as isize, 1.0, c, O as isize, 1,
        )
    };
}

fn vecvec_mul_into<const M: usize, const O: usize>(
    x: &[f32; M],
    y: &[f32; O],
    out: &mut [[f32; O]; M],
) {
    unsafe {
        let a = x.as_ptr() as *const f32;
        let b = y.as_ptr() as *const f32;
        let c = out.as_mut_ptr() as *mut f32;
        sgemm(
            M, 1, O, 1.0, a, 1, 1, b, O as isize, 1, 1.0, c, O as isize, 1,
        )
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vecmul() {
        let x = [1.0, 2.0, 3.0];
        let y = [[1.0, 2.0], [0.5, 1.0], [1.0 / 3.0, 1.0]];
        let y_t = [[1.0, 0.5, 1.0 / 3.0], [2.0, 1.0, 1.0]];
        let expected = [3.0, 7.0];

        let mut out = [0.0; 2];
        vecmat_mul_into(&x, &y, &mut out);
        assert_eq!(out, expected);

        let mut out = [0.0; 2];
        vecmat_mul_into_yt(&x, &y_t, &mut out);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_matmul() {
        let x = [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ];
        let x_t = [
            [1.0, 4.0, 7.0, 10.0],
            [2.0, 5.0, 8.0, 11.0],
            [3.0, 6.0, 9.0, 12.0],
        ];
        let y = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let y_t = [[1.0, 3.0, 5.0], [2.0, 4.0, 6.0]];
        let expected = [[22.0, 28.0], [49.0, 64.0], [76.0, 100.0], [103.0, 136.0]];

        let mut out = [[0.0; 2]; 4];
        matmat_mul_into(&x, &y, &mut out);
        assert_eq!(out, expected);

        let mut out = [[0.0; 2]; 4];
        matmat_mul_into_xt(&x_t, &y, &mut out);
        assert_eq!(out, expected);

        let mut out = [[0.0; 2]; 4];
        matmat_mul_into_yt(&x, &y_t, &mut out);
        assert_eq!(out, expected);
    }

    #[test]
    fn test_vecvec() {
        let x = [1.0, 2.0, 3.0];
        let y = [-1.0, 0.5, -1.0 / 3.0, 0.25];

        let mut out = [[0.0; 4]; 3];
        vecvec_mul_into(&x, &y, &mut out);
        assert_eq!(
            out,
            [
                [-1.0, 0.5, -1.0 / 3.0, 0.25],
                [-2.0, 1.0, -2.0 / 3.0, 0.5],
                [-3.0, 1.5, -1.0, 0.75],
            ]
        );

        let mut out = [[0.0; 3]; 4];
        vecvec_mul_into(&y, &x, &mut out);
        assert_eq!(
            out,
            [
                [-1.0, -2.0, -3.0],
                [0.5, 1.0, 1.5],
                [-1.0 / 3.0, -2.0 / 3.0, -1.0],
                [0.25, 0.5, 0.75],
            ]
        );
    }

    #[test]
    fn test_matmat_mul() {
        let a = Tensor2D::new([
            [0.5086, 0.5234, 0.2684],
            [0.8075, 0.8437, 0.9951],
            [0.0774, 0.7539, 0.8894],
            [0.8119, 0.2693, 0.7249],
        ]);
        let b = Tensor2D::new([[0.4651, 0.9106], [0.3360, 0.5534], [0.8092, 0.3827]]);
        let r: Tensor2D<4, 2, OwnedTape> = matmul(a.trace(), &b);
        assert_eq!(
            r.data(),
            &[
                [0.62960154, 0.8554974],
                [1.4642863, 1.5830379],
                [1.0090116, 0.82806206],
                [1.0546886, 1.165766]
            ]
        );
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[
                [0.37689444, 0.24156547, 0.30238447],
                [0.80570966, 0.5184905, 0.6703743],
                [0.4199963, 0.2735345, 0.38693744],
                [0.5321113, 0.34252504, 0.4438907]
            ]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[
                [0.8737376, 0.9888564],
                [0.9339924, 0.991189],
                [1.1659734, 1.2298465]
            ]
        );
    }

    #[test]
    fn test_matmul_transpose() {
        let a = Tensor2D::new([
            [0.5086, 0.5234, 0.2684],
            [0.8075, 0.8437, 0.9951],
            [0.0774, 0.7539, 0.8894],
            [0.8119, 0.2693, 0.7249],
        ]);
        let b = Tensor2D::new([[0.4651, 0.3360, 0.8092], [0.9106, 0.5534, 0.3827]]);
        let r: Tensor2D<4, 2, OwnedTape> = matmul_transpose(a.trace(), &b);
        assert_eq!(
            r.data(),
            &[
                [0.62960154, 0.8554974],
                [1.4642863, 1.5830379],
                [1.0090116, 0.82806206],
                [1.0546886, 1.165766]
            ]
        );
        let gradients = r.mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[[0.1719625, 0.111175, 0.1489875]; 4]
        );
    }

    #[test]
    fn test_vecmat_mul() {
        let a = Tensor1D::new([0.7296, 0.3974, 0.9487]);
        let b = Tensor2D::new([[0.7804, 0.5540], [0.5378, 0.8401], [0.5042, 0.8604]]);
        let r: Tensor1D<2, OwnedTape> = vecmat_mul(a.trace(), &b);
        assert_eq!(r.data(), &[1.261436, 1.5543157]);
        let gradients = r.exp().mean().backward();
        assert_eq!(
            gradients.ref_gradient(&a),
            &[2.6883178, 2.9369607, 2.9256766]
        );
        assert_eq!(
            gradients.ref_gradient(&b),
            &[
                [1.2879219, 1.7261779],
                [0.70150787, 0.94021803],
                [1.6746868, 2.244552]
            ]
        );
    }

    #[test]
    fn test_vecmat_mul_transpose() {
        let a = Tensor1D::new([0.7296, 0.3974, 0.9487]);
        let b = Tensor2D::new([[0.7804, 0.5378, 0.5042], [0.5540, 0.8401, 0.8604]]);
        let r: Tensor1D<2, OwnedTape> = vecmat_mul_transpose(a.trace(), &b);
        assert_eq!(r.data(), &[1.261436, 1.5543157]);
        let gradients = r.mean().backward();
        assert_eq!(gradients.ref_gradient(&a), &[0.66719997, 0.68895, 0.6823]);
    }
}

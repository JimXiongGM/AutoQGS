import math
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

import utils

# ---------- my ---------- #


"""
adopt from allennlp 2.8.0
"""


class Attention(torch.nn.Module):
    """
    An `Attention` takes two inputs: a (batched) vector and a matrix, plus an optional mask on the
    rows of the matrix.  We compute the similarity between the vector and each row in the matrix,
    and then (optionally) perform a softmax over rows using those computed similarities.


    Inputs:

    - vector: shape `(batch_size, embedding_dim)`
    - matrix: shape `(batch_size, num_rows, embedding_dim)`
    - matrix_mask: shape `(batch_size, num_rows)`, specifying which rows are just padding.

    Output:

    - attention: shape `(batch_size, num_rows)`.

    # Parameters

    normalize : `bool`, optional (default = `True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, normalize: bool = True) -> None:
        super().__init__()
        self._normalize = normalize

    def forward(
        self,
        vector: torch.Tensor,
        matrix: torch.Tensor,
        matrix_mask: torch.BoolTensor = None,
    ) -> torch.Tensor:
        similarities = self._forward_internal(vector, matrix)
        if self._normalize:
            return utils.masked_softmax(similarities, matrix_mask)
        else:
            return similarities

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


# ---------- basic attentions ---------- #


class BilinearAttention(Attention):
    """
    Computes attention between a vector and a matrix using a bilinear attention function.  This
    function has a matrix of weights `W` and a bias `b`, and the similarity between the vector
    `x` and the matrix `y` is computed as `x^T W y + b`.

    Registered as an `Attention` with name "bilinear".

    # Parameters

    vector_dim : `int`, required
        The dimension of the vector, `x`, described above.  This is `x.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_dim : `int`, required
        The dimension of the matrix, `y`, described above.  This is `y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    activation : `Activation`, optional (default=`linear`)
        An activation function applied after the `x^T W y + b` calculation.  Default is
        linear, i.e. no activation.
    normalize : `bool`, optional (default=`True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(
        self,
        vector_dim: int,
        matrix_dim: int,
        activation=None,
        normalize: bool = True,
    ) -> None:
        super().__init__(normalize)
        self._weight_matrix = Parameter(torch.Tensor(vector_dim, matrix_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation or (lambda x: x)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._weight_matrix)
        self._bias.data.fill_(0)

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        calculation:
            `x^T W y + b`
        intuition:
            - convert vector dim to matrix dim and use matrix multiplication to calculate similarity of vector and each row of matrix.

        inputs:
            vector: [bsz, vector_dim]
            matrix: [bsz, matrix_len, matrix_dim]
        returns:
            res: [bsz, matrix_len]
        """
        # [bsz, 1, matrix_dim]
        intermediate = vector.mm(self._weight_matrix).unsqueeze(1)

        # [bsz, matrix_len] = [bsz, 1, matrix_dim] @ [bsz, matrix_dim, matrix_len] + [bsz, 1]
        res = self._activation(intermediate.bmm(matrix.transpose(1, 2)).squeeze(1) + self._bias)
        return res


class AdditiveAttention(Attention):
    """
    Computes attention between a vector and a matrix using an additive attention function.  This
    function has two matrices `W`, `U` and a vector `V`. The similarity between the vector
    `x` and the matrix `y` is computed as `V tanh(Wx + Uy)`.

    This attention is often referred as concat or additive attention. It was introduced in
    [Neural Machine Translation by Jointly Learning to Align and Translate (Bahdanau et al, 2015)]
    (https://api.semanticscholar.org/CorpusID:11212020).

    Registered as an `Attention` with name "additive".

    # Parameters

    vector_dim : `int`, required
        The dimension of the vector, `x`, described above.  This is `x.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    matrix_dim : `int`, required
        The dimension of the matrix, `y`, described above.  This is `y.size()[-1]` - the length
        of the vector that will go into the similarity computation.  We need this so we can build
        the weight matrix correctly.
    normalize : `bool`, optional (default = `True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, vector_dim: int, matrix_dim: int, normalize: bool = True) -> None:
        super().__init__(normalize)
        self._w_matrix = Parameter(torch.Tensor(vector_dim, vector_dim))
        self._u_matrix = Parameter(torch.Tensor(matrix_dim, vector_dim))
        self._v_vector = Parameter(torch.Tensor(vector_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self._w_matrix)
        torch.nn.init.xavier_uniform_(self._u_matrix)
        torch.nn.init.xavier_uniform_(self._v_vector)

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        calculation:
            `V tanh(Wx + Uy)`
        intuition:
            - broadcast vector to matrix shape and add both, then use linear to reduce dim to 1.

        inputs:
            vector: [bsz, vector_dim]
            matrix: [bsz, matrix_len, matrix_dim]
        returns:
            res: [bsz, matrix_len]
        """
        # [bsz, matrix_len, vector_dim]
        intermediate = vector.matmul(self._w_matrix).unsqueeze(1) + matrix.matmul(self._u_matrix)
        intermediate = torch.tanh(intermediate)

        # [bsz, matrix_len] = [bsz, matrix_len, vector_dim] @ [bsz, vector_dim, 1]
        res = intermediate.matmul(self._v_vector).squeeze(2)
        return res


class CosineAttention(Attention):
    """
    Computes attention between a vector and a matrix using cosine similarity.

    Registered as an `Attention` with name "cosine".
    """

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        calculation:
            `norm(a) @ norm(b)`
        intuition:
            - matrix multiplication.

        inputs:
            vector: [bsz, dim]
            matrix: [bsz, matrix_len, dim]
        returns:
            res: [bsz, matrix_len]
        """
        a_norm = vector / (vector.norm(p=2, dim=-1, keepdim=True) + utils.tiny_value_of_dtype(vector.dtype))
        b_norm = matrix / (matrix.norm(p=2, dim=-1, keepdim=True) + utils.tiny_value_of_dtype(matrix.dtype))
        res = torch.bmm(a_norm.unsqueeze(dim=1), b_norm.transpose(-1, -2)).squeeze(1)
        return res


# ---------- DotProduct ---------- #


class DotProductAttention(Attention):
    """
    Computes attention between a vector and a matrix using dot product.

    Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    (https://api.semanticscholar.org/CorpusID:13756489)

    Registered as an `Attention` with name "dot_product".
    """

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        calculation:
            `a @ b`
        intuition:
            - matrix multiplication.

        inputs:
            vector: [bsz, dim]
            matrix: [bsz, matrix_len, dim]
        returns:
            res: [bsz, matrix_len]
        """
        return matrix.bmm(vector.unsqueeze(-1)).squeeze(-1)


class ScaledDotProductAttention(DotProductAttention):
    """
    Computes attention between two tensors using scaled dot product.
    # Reference: [Attention Is All You Need (Vaswani et al, 2017)]
    # (https://api.semanticscholar.org/CorpusID:13756489)

    Registered as an `Attention` with name "scaled_dot_product".

    # Parameters

    scaling_factor : `int`, required
        The similarity score is scaled down by the `scaling_factor`.
    normalize : `bool`, optional (default=`True`)
        If true, we normalize the computed similarities with a softmax, to return a probability
        distribution for your attention.  If false, this is just computing a similarity score.
    """

    def __init__(self, scaling_factor: Optional[int] = None, normalize: bool = True) -> None:
        super().__init__(normalize)
        self.scaling_factor = scaling_factor

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        scores = super()._forward_internal(vector, matrix)
        scaling_factor = self.scaling_factor or matrix.size(-1)
        scores = scores / math.sqrt(scaling_factor)
        return scores


# ---------- complex ---------- #


class LinearAttention(Attention):
    """
    This `Attention` module performs a dot product between a vector of weights and some
    combination of the two input vectors, followed by an (optional) activation function.  The
    combination used is configurable.

    If the two vectors are `x` and `y`, we allow the following kinds of combinations : `x`,
    `y`, `x*y`, `x+y`, `x-y`, `x/y`, where each of those binary operations is performed
    elementwise.  You can list as many combinations as you want, comma separated.  For example, you
    might give `x,y,x*y` as the `combination` parameter to this class.  The computed similarity
    function would then be `w^T [x; y; x*y] + b`, where `w` is a vector of weights, `b` is a
    bias parameter, and `[;]` is vector concatenation.

    Note that if you want a bilinear similarity function with a diagonal weight matrix W, where the
    similarity function is computed as `x * w * y + b` (with `w` the diagonal of `W`), you can
    accomplish that with this class by using "x*y" for `combination`.

    Registered as an `Attention` with name "linear".

    # Parameters

    tensor_1_dim : `int`, required
        The dimension of the first tensor, `x`, described above.  This is `x.size()[-1]` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    tensor_2_dim : `int`, required
        The dimension of the second tensor, `y`, described above.  This is `y.size()[-1]` - the
        length of the vector that will go into the similarity computation.  We need this so we can
        build weight vectors correctly.
    combination : `str`, optional (default=`"x,y"`)
        Described above.
    activation : `Activation`, optional (default=`linear`)
        An activation function applied after the `w^T * [x;y] + b` calculation.  Default is
        linear, i.e. no activation.
    normalize : `bool`, optional (default=`True`)
    """

    def __init__(
        self,
        tensor_1_dim: int,
        tensor_2_dim: int,
        combination: str = "x,y",
        activation=None,
        normalize: bool = True,
    ) -> None:
        """
        notes:
            - if specify `x*y` in combination, then `tensor_1_dim` must equal to `tensor_2_dim`.

        get_combined_dim 说明
            - 构建目标张量需要的维度，例如`x,y`表示目标张量的dim是x_dim+y_dim，`x,x,y`表示x_dim++x_dim+y_dim，就是在dim=-1维度cat张量，这里不需要x和y的维度相同。
            - 加减乘除操作是element wise的，因此x和y在dim=-1的维度必须相同
        """
        super().__init__(normalize)
        self._combination = combination

        # cal the dim satisfied all combinations.
        combined_dim = utils.get_combined_dim(combination, [tensor_1_dim, tensor_2_dim])
        self._weight_vector = Parameter(torch.Tensor(combined_dim))
        self._bias = Parameter(torch.Tensor(1))
        self._activation = activation or (lambda x: x)
        self.reset_parameters()

    def reset_parameters(self):
        std = math.sqrt(6 / (self._weight_vector.size(0) + 1))
        self._weight_vector.data.uniform_(-std, std)
        self._bias.data.fill_(0)

    def _forward_internal(self, vector: torch.Tensor, matrix: torch.Tensor) -> torch.Tensor:
        """
        combine_tensors_and_multiply 说明
        - 计算向量x和矩阵y的相似度，返回 [bsz, matrix_len]
        - 此函数根据 combination 构建了一个大的 _weight_vector 然后计算每个 combination 的得分（标量），最后相加。
        - 例如 [x,y,x*y] 先构建dim*3的 _weight_vector: [dim for x; dim for y; dim for x*y] 分成3段，x:[bsz, dim]  y: [bsz, y_len, dim] ，然后 x @ _weight_vector[dim,1] = [bsz, 1] ，y部分同理 y @ _weight_vector[dim,1] = [bsz, y_len, 1] ，x*y的shape和y一致，最后直接把3个标量相加即可。

        inputs:
            vector: [bsz, dim]
            matrix: [bsz, matrix_len, dim]

        returns:
            res: [bsz, matrix_len]
        """
        combined_tensors = utils.combine_tensors_and_multiply(
            combination=self._combination,
            tensors=[vector.unsqueeze(1), matrix],
            weights=self._weight_vector,
        )
        res = self._activation(combined_tensors.squeeze(1) + self._bias)
        return res


class AttentionWithCoverage(nn.Module):
    """
    attention module for coverage mechanism in decoder.
    """

    def __init__(self, vector_dim: int, matrix_dim: int, normalize: bool = True) -> None:
        super(AttentionWithCoverage, self).__init__()
        self._normalize = normalize
        self.W1 = nn.Linear(1, vector_dim, bias=False)
        self._w_matrix = nn.Parameter(torch.Tensor(vector_dim, vector_dim))
        self._u_matrix = nn.Parameter(torch.Tensor(matrix_dim, vector_dim))
        self._v_vector = nn.Parameter(torch.Tensor(vector_dim, 1))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self._w_matrix)
        nn.init.xavier_uniform_(self._u_matrix)
        nn.init.xavier_uniform_(self._v_vector)

    def _forward_internal(
        self, vector: torch.Tensor, matrix: torch.Tensor, coverage: torch.Tensor = None
    ) -> torch.Tensor:
        """
        calculation:
            `V tanh(Wx + Uy)`
        intuition:
            - broadcast vector to matrix shape and add both, then use linear to reduce dim to 1.

        inputs:
            vector: [bsz, vector_dim]
            matrix: [bsz, matrix_len, matrix_dim]
        returns:
            res: [bsz, matrix_len]
        """
        # [bsz, matrix_len, vector_dim]
        intermediate = vector.matmul(self._w_matrix).unsqueeze(1) + matrix.matmul(self._u_matrix)
        if coverage is not None:
            intermediate = intermediate + self.W1(coverage.unsqueeze(2))

        intermediate = torch.tanh(intermediate)

        # [bsz, matrix_len] = [bsz, matrix_len, vector_dim] @ [bsz, vector_dim, 1]
        res = intermediate.matmul(self._v_vector).squeeze(2)
        return res

    def forward(
        self,
        vector: torch.Tensor,
        matrix: torch.Tensor,
        matrix_mask: torch.BoolTensor = None,
        coverage: torch.Tensor = None,
    ) -> torch.Tensor:
        similarities = self._forward_internal(vector, matrix, coverage)
        if self._normalize:
            similarities = utils.masked_softmax(similarities, matrix_mask)

        # [bsz, hid]
        weighted_enc_hiddens = torch.matmul(similarities.unsqueeze(1), matrix).squeeze(1)
        if coverage is not None:
            coverage = coverage + similarities

        if True in torch.isnan(weighted_enc_hiddens):
            x = 1

        if True in torch.isnan(similarities):
            x = 1

        return weighted_enc_hiddens, similarities, coverage


def test01():
    # settings 1
    bsz = 3
    vector_dim = 49
    matrix_len = 7
    matrix_dim = 51
    vector = torch.rand(bsz, vector_dim)
    matrix = torch.rand(bsz, matrix_len, matrix_dim)
    attn = LinearAttention(
        tensor_1_dim=vector_dim,
        tensor_2_dim=matrix_dim,
        combination="x,x,y",  # [x; y; x*y; x+y, x-y; x/y]
        activation=torch.relu,
    )
    res1 = attn(vector, matrix)

    # settings 2
    bsz = 3
    vector_dim = 51
    matrix_len = 7
    matrix_dim = 51
    vector = torch.rand(bsz, vector_dim)
    matrix = torch.rand(bsz, matrix_len, matrix_dim)
    attn = LinearAttention(
        tensor_1_dim=vector_dim,
        tensor_2_dim=matrix_dim,
        combination="x,y,x*y",  # [x; y; x*y; x+y, x-y; x/y]
        activation=torch.relu,
    )
    # res: [3,3,7] ??
    res2 = attn(vector, matrix)
    _ = 1

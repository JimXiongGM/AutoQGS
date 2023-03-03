def Levenshtein_Distance(str1, str2):
    """
    Calculate the edit distance between strings str1 and str2
    :param str1
    :param str2
    :return:
    """
    matrix = [[i + j for j in range(len(str2) + 1)] for i in range(len(str1) + 1)]

    for i in range(1, len(str1) + 1):
        for j in range(1, len(str2) + 1):
            if str1[i - 1] == str2[j - 1]:
                d = 0
            else:
                d = 1

            matrix[i][j] = min(matrix[i - 1][j] + 1, matrix[i][j - 1] + 1, matrix[i - 1][j - 1] + d)

    return matrix[len(str1)][len(str2)]


class KMP:
    """
    KMP algorithm, returns matching substrings (continuous subsequences).
    from: https://zhuanlan.zhihu.com/p/41047378
    usage:
        KMP()(["it", "is", "hello", "world"], ["hello", "world"])
    """

    def kmp(self, main_str, sub_str):
        """
        main_str is the main string, sub_str is the pattern string, if sub_str is in t, returns the first occurrence index
        main_str, sub_str can be str or list
        """
        nex = self.getNext(sub_str)
        i = 0
        j = 0
        while i < len(main_str) and j < len(sub_str):
            if j == -1 or main_str[i] == sub_str[j]:
                i += 1
                j += 1
            else:
                j = nex[j]

        if j == len(sub_str):
            return i - j
        else:
            return -1

    def getNext(self, sub_str):
        """
        sub_str is the pattern string
        returns the next array, i.e., the partial match table
        """
        nex = [0] * (len(sub_str) + 1)
        nex[0] = -1
        i = 0
        j = -1
        while i < len(sub_str):
            if j == -1 or sub_str[i] == sub_str[j]:
                i += 1
                j += 1
                nex[i] = j
            else:
                j = nex[j]

        return nex

    @staticmethod
    def replace_substr(main_str, sub_str, start_idx, replace):
        if start_idx == -1:
            return main_str
        if isinstance(main_str, list):
            replace = [replace] if type(replace) == str else replace
        new_main_str = main_str[:start_idx] + replace + main_str[start_idx + len(sub_str) :]
        return new_main_str

    def __call__(self, main_str, sub_str, replace=None):
        """
        inputs:
            sub: ["hello", "world"]
            string: ["it", "is", "hello", "world"]
            replace: "ent_001"
        return:
            ["it", "is", "ent_001"]
        """
        assert (isinstance(main_str, str) and isinstance(sub_str, str)) or (
            isinstance(main_str, list) and isinstance(sub_str, list)
        ), f"wrong type: {type(main_str)} and {type(sub_str)} must be the same (both str or list)."
        start_idx = self.kmp(main_str, sub_str)
        if replace is None:
            return start_idx
        new_main_str = KMP.replace_substr(main_str, sub_str, start_idx, replace)
        return new_main_str

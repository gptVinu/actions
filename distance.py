def calculate_edit_distance_recursive(str1, str2, m, n):
    """Calculates edit distance recursively."""
    if m == 0:
        return n
    if n == 0:
        return m
    if str1[m - 1] == str2[n - 1]:
        return calculate_edit_distance_recursive(str1, str2, m - 1, n - 1)
    return 1 + min(
        calculate_edit_distance_recursive(str1, str2, m, n - 1),
        calculate_edit_distance_recursive(str1, str2, m - 1, n),
        calculate_edit_distance_recursive(str1, str2, m - 1, n - 1)
    )

def calculate_edit_distance_dp(str1, str2):
    """Calculates edit distance using dynamic programming (Levenshtein)."""
    m, n = len(str1), len(str2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i][j - 1], dp[i - 1][j], dp[i - 1][j - 1])
    return dp[m][n]

if __name__ == "__main__":
    print("Calculating Edit/Levenshtein Distances...")
    string1 = input("Enter string 1: ")
    string2 = input("Enter string 2: ")

    recursive_distance = calculate_edit_distance_recursive(string1, string2, len(string1), len(string2))
    dp_distance = calculate_edit_distance_dp(string1, string2)

    print(f"Recursive Edit Distance: {recursive_distance}")
    print(f"Levenshtein Distance (Dynamic Programming): {dp_distance}")
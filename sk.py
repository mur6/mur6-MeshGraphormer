from sklearn.linear_model import LinearRegression

# グラフ描画ライブラリ Matplotlib を使用する


# アンスコムのデータセット
X = [[5.566], [3.93], [5.2276855], [5.88824], [5.185482]]
y = [-0.38, -0.495, -0.43, -0.315, -0.456]
# 最小二乗法モデルで予測式を求める
model = LinearRegression()
model.fit(X, y)
print('切片:', model.intercept_)
print('傾き:', model.coef_)


# 6.2047143	-0.429
# 5.185482	-0.456
# 7.3168364	-0.301

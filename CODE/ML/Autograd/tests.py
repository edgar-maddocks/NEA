from Tensor import Tensor

a = Tensor(1, requires_grad=True)
b = Tensor(2, requires_grad=True)

a.zero_grad()
y = a + b
y.backward()
assert a.grad == 1

a.zero_grad()
y = a + 2
y.backward()
assert a.grad == 1

a.zero_grad()
a += 2
a.backward()
assert a.grad == 1

print("PASSED")

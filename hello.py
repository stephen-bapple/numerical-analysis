print("This program finds the roots of a quadratic.")
print("Such as ax^2 + b + c")

# ax^2 + bx + c 

a = int(input("Enter a: "))
b = int(input("Enter b: "))
c = int(input("Enter c: "))

root1 = (-b + (b**2 - 4 * a * c)**(1/2)) / (2 * a)

root2 = (-b - (b**2 - 4 * a * c)**(1/2)) / (2 * a)

print("The roots are ", root1, "and", root2)
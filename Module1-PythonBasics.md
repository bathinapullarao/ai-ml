# Learning Objectives
``` bash
By the end of this module, learners will be able to:

1.Explain and apply operators in Python for performing arithmetic, relational, logical, and bitwise operations.
2.Implement decision-making constructs (such as if, elif, and else) to control program flow based on conditions.
3.Use looping structures (for and while loops) to execute repetitive tasks efficiently.
4.Create and manipulate lists for storing and processing ordered collections of data.
5.Work with tuples to handle immutable sequences in Python.
6.Utilize dictionaries for key-value data storage and retrieval.
7.Define and call functions to organize code into reusable blocks.
8.Perform file input and output (I/O) operations to read from and write to external files.
9.Import and use modules to extend Python functionality and organize code.
10.Handle exceptions using try-except blocks to make programs robust and error-tolerant.
```
# What is Python?
``` bash
Python is a popular programming language. It was created by Guido van Rossum, and released in 1991.

It is used for:
web development (server-side),
software development and
system scripting.
```
# What can Python do?
``` bash
Python can be used on a server to create web applications.
Python can be used alongside software to create workflows.
Python can connect to database systems. It can also read and modify files.
Python can be used to handle big data and perform complex mathematics.
Python can be used for rapid prototyping, or for production-ready software development.
```

# Why Python?
``` bash
Python works on different platforms (Windows, Mac, Linux, Raspberry Pi, etc).
Python has a simple syntax similar to the English language.
Python has syntax that allows developers to write programs with fewer lines than some other programming languages.
Python runs on an interpreter system, meaning that code can be executed as soon as it is written. This means that prototyping can be very quick.
Python can be treated in a procedural way, an object-oriented way or a functional way.
```

# Executing a Python Program
``` bash
Python is an interpreted programming language, this means that as a developer you write Python (.py) files in a text editor and then put those files
into the python interpreter to be executed. You can also execute your Python program using vscode, anaconda navigator and google colab. 
```
# Hello World Program
``` bash
A computer program is a list of "instructions" to be "executed" by a computer. In a programming language, these programming instructions are called statements. The following statement prints the text "Hello World" to the screen:

print("Hello World!")

In Python, a statement usually ends when the line ends. You do not need to use a semicolon (;) like in many other programming languages.
Most Python programs contain many statements. The statements are executed one by one, in the same order as they are written:

print("Hello World!")
print("Have a good day.")
print("Learning Python is fun!")

Text in Python must be inside quotes. You can use either " double quotes or ' single quotes.
```
# Variables
``` bash
In Python, variables are used to store data that can be referenced and manipulated during program execution. A variable is essentially a name that is assigned to a value.

Unlike Java and many other languages, Python variables do not require explicit declaration of type.
The type of the variable is inferred based on the value assigned.
```
# Creating Variables
``` bash
To create a variable in Python, you simply choose a name and use the assignment operator (=) to assign a value to it. For example, if you want to create a variable named "age" and assign it the value 25, you would write:

age = 25

Python is dynamically typed, meaning you do not need to declare the variable type before using it. The type is inferred from the value assigned. This allows for flexibility, as a variable can hold different types of data at different times. For instance, you could later assign a string to the same variable:

age = "twenty-five"

In Python, variables can hold various data types, including integers, floats, strings, lists, tuples, dictionaries, and more.
Naming conventions for variables generally suggest using descriptive names that reflect the data they store,
such as "user_name" or "total_price".

It is also worth noting that Python variables are case-sensitive, meaning "Variable" and "variable" would be treated as two distinct identifiers. Additionally, certain keywords in Python, like "if" or "for", cannot be used as variable names.
```
# Rules for naming Variables
``` bash
variable naming conventions:
In Python, variable names must start with a letter (a-z, A-Z) or an underscore (_). They can be followed by letters, digits (0-9), or underscores.
This means you cannot start a variable name with a number.

case Sensitivity:
Variable names in Python are case-sensitive. This means that 'myVar', 'MyVar', and 'MYVAR' are all considered different variables.
It's essential to be consistent with your casing to avoid confusion.

Avoiding reserved words:
You should not use Python's reserved keywords as variable names. Words like 'class', 'for', 'if', and 'while' have special meanings in Python
and using them can lead to syntax errors.

Meaningful names:
Choosing meaningful variable names is a good practice. Instead of using generic names like 'x' or 'temp', opt for descriptive names like
'user_age' or 'total_price' to make your code more understandable.

using underscores:
For variable names that consist of multiple words, use underscores to separate them, such as 'first_name' or 'total_count'.
This improves readability and aligns with the PEP 8 style guide.

Length of variable name:
While there is no strict limit on the length of variable names, it's advisable to keep them reasonably short while still being descriptive.
This helps maintain readability and avoids clutter.

Global vs Local variables:
When naming variables, be mindful of their scope. Global variables should be named with a clear indication of their purpose,
while local variables can be more concise but still meaningful.

Usning constants:
For constants, it's a common practice to use all uppercase letters with underscores separating words, like 'MAX_VALUE' or 'PI_VALUE'.
This signals to other developers that these values should not be changed.
```
# Assigning values to variables
``` bash
Assigning values to variables in Python is a fundamental concept that allows programmers to store and manipulate data. In Python,
a variable is created by simply assigning a value to a name using the equals sign. 

For example, to create a variable named `age` and assign it the value of 25, you would write `age = 25`. 



Python is dynamically typed, which means that you do not need to declare the type of a variable explicitly. The interpreter
 automatically determines the type based on the value assigned to it. 

For instance, if you assign a string value like `name = "Alice"`, Python understands that `name` is a string.

You can also change the value of a variable at any time. If later in your code you wanted to update `age` to 30, you would simply write
`age = 30`. This flexibility allows for dynamic data manipulation throughout a program.

Furthermore, you can assign multiple variables simultaneously. 

For example, `x, y, z = 1, 2, 3` assigns the value 1 to `x`, 2 to `y`, and 3 to `z` in a single line. 
```
# Type Casting
``` bash
Type casting in Python refers to the conversion of one data type into another. This process can be necessary when you need to perform
operations on different types of data or when you're working with data inputs that may not be in the expected format.

1. Implicit type casting, also known as coercion, occurs when Python automatically converts one data type to another without explicit
instruction from the programmer. This often happens in operations that involve mixed data types, such as adding an integer to a float,
where the integer is automatically converted to a float.

2.Explicit type casting, on the other hand, is when you manually change the type of a variable using Python's built-in functions.
Common functions for explicit casting include int(), float(), and str(), allowing you to specify the desired type clearly.

When receiving user input through the input() function, the data is captured as a string by default. Therefore, if you need to perform arithmetic operations on that input, you must explicitly cast it to the desired type, such as int or float.
```
# Operators
``` bash
Operators in Python are special symbols that perform operations on variables and values.
# Types of Operators in Python
1. Arithmetic operators are used to perform mathematical operations such as addition, subtraction, multiplication, and division.
In Python, the basic arithmetic operators include + for addition, - for subtraction, * for multiplication, / for division, and % for modulus.

2.Comparison operators are used to compare two values. They return a Boolean value (True or False) based on the comparison.
Common comparison operators in Python include == for equality, != for inequality, > for greater than, < for less than, >= for
greater than or equal to, and <= for less than or equal to.

3.Logical operators are used to combine conditional statements. They include 'and', 'or', and 'not'. The 'and' operator returnsTrue if both
statements are true, while 'or' returns True if at least one statement is true. The 'not' operator reverses the result of the condition.

4.Bitwise operators are used to perform operations on binary numbers at the bit level. They include operators like &, |, ^, ~, <<, and >>.
These operators allow manipulation of individual bits within integers, enabling more efficient programming in certain scenarios.

5.Assignment operators are used to assign values to variables. The basic assignment operator is '=', but Python also includes compound
assignment operators like += for addition and assignment, -= for subtraction and assignment, and *= for multiplication and assignment.

6.Identity operators are used to check if two values are located on the same part of memory. The 'is' operator returns True if both
operands refer to the same object, while 'is not' returns True if they do not.

7.Membership operators are used to test if a value is part of a sequence, such as a list, tuple, or string. The 'in' operator returns
True if the value is found in the sequence, while 'not in' returns True if it is not found.

8.The ternary operator in Python is a shorthand for the if-else statement. It allows for a concise way to return one of two values
based on a condition. The syntax is 'value_if_true if condition else value_if_false'.
```
# Comments
``` bash
In Python, comments are crucial for documenting code effectively. They help clarify the logic behind code sections and make collaboration easier.
Comments can be added using the hash symbol (#), which allows you to write notes alongside your code without affecting its execution.
This practice not only aids in personal understanding but also provides context for others who may work with the code in the future.
Using clear and concise comments can greatly enhance readability and maintainability of the code.

# How to Add Comments in Python

For single-line comments, placing a # symbol in front of the text is sufficient. For multi-line comments, triple quotes can be used,
allowing you to include longer explanations or notes without interrupting the flow of the code. Both practices significantly contribute
to code readability and maintainability.
```
# Input and Output in Python
``` bash
Input and output statements are essential for interacting with users and displaying information. The primary function for outputting data
is the print() function. It takes one or more arguments, formats them if necessary, and displays them on the screen. You can also format
the output using various methods, such as f-strings or the format() method, to create more readable and structured output.

For input, Python provides the input() function, which allows users to enter data through the console. The input collected is always
returned as a string, so if you need a different data type, you’ll need to convert it using functions like int(), float(), etc. You
can also provide a prompt within the input() function to guide the user on what to enter.

Effective programming with input and output involves understanding how to manage user input properly, validate it, and handle errors
gracefully. This ensures a better user experience and minimizes the risk of program crashes due to unexpected input.

The 'input()' Function

The input() function in Python is a built-in function that allows for user input. It reads a line from input, typically from the keyboard, and returns it as a string after the user presses Enter. This function is commonly used for interactive programs where user interaction is needed. 

Syntax:
The basic syntax of the input() function is as follows:

input([prompt])

Here, 'prompt' is an optional string that is displayed to the user before reading the input. If provided, it serves as a message to guide the user about what kind of input is expected.
For instance, asking for their name through the program.

name = input("Enter your name: ")
print("Hello, " + name)

While input() is straightforward, it's worth noting that it always returns data as a string. If numeric input is required, the input must be converted to the appropriate type, such as using int() or float().

The 'print()' Function

The print() function in Python is used to output data to the console. Its basic syntax is as follows:

print(object(s), sep=' ', end='\n', file=sys.stdout, flush=False)

- object(s): This is the value or values you want to print. You can pass multiple objects separated by commas.
- sep: This is an optional parameter that specifies the string to be inserted between the objects. The default is a space.
- end: This defines the string that is printed at the end of the output. The default is a newline character.
- file: This is an optional parameter that specifies where to send the output. By default, it goes to sys.stdout.
- flush: This is a boolean parameter that indicates whether to flush the output buffer. The default is False.

To format output, you can use f-strings, format() method, or the % operator.Using f-strings, you can directly embed expressions
inside string literals. 

For example:

name = "Alice"
age = 30
print(f"My name is {name} and I am {age} years old.")

Using the format() method, you can achieve similar results:

print("My name is {} and I am {} years old.".format(name, age))

Alternatively, using the % operator for string formatting:

print("My name is %s and I am %d years old." % (name, age))
```

# What is a datatype
``` bash
A datatype is a classification that specifies which type of value a variable can hold in programming and computer science.
Each datatype has its own set of rules and behaviors, impacting how data is manipulated within a program. Understanding
datatypes is essential for effective programming, as it influences performance, memory usage, and the overall logic of the code.

Common Data Types

Some common data types include:

1. Integer: Represents whole numbers, both positive and negative.
2. Float: Represents decimal numbers, allowing for fractional values.
3. String: A sequence of characters, used to represent text.
4. Boolean: Represents true or false values, often used in conditional statements.
5. List: A collection of items that can store multiple values, typically of varying data types.
6. Dictionary: A collection of key-value pairs, where each key is unique.
7. Set: A collection of unique items, often used for membership testing and eliminating duplicate entries.

Exploring Strings

Strings are data types used to represent text. They are enclosed in either single or double quotes and can contain letters,
numbers, symbols, and spaces. Strings can be manipulated using various functions such as concatenation, slicing, and formatting.

String operations 

String operations are essential for manipulating and working with text data. Python provides a variety of built-in methods that
make it easy to modify and interact with strings.One of the most common operations is string concatenation, which is accomplished
using the plus operator. For example, you can combine two strings with "Hello" + " World" to get "Hello World". Another important
operation is string slicing, which allows you to access a substring by specifying a range of indices. For instance, if you have a
string "Python", slicing it with string[0:2] will give you "Py".


You can also use various methods to modify strings. The `upper()` method converts all characters in a string to uppercase, while
`lower()` changes them to lowercase. The `strip()` method removes leading and trailing whitespace, and `replace()` allows you to
replace a specific substring with another substring. Searching within strings can be done using methods like `find()` and `count()`.
The `find()` method returns the index of the first occurrence of a substring, and `count()` returns the number of times a substring
appears in the string.


Strings in Python are immutable, meaning they cannot be modified after creation. Operations that appear to modify a string actually
return a new string. For example, when you use the `replace()` method, it returns a new string with the replacements made, leaving the
original string unchanged.

Finally, string formatting provides a way to create new strings by embedding variables within them. The `format()` method and f-strings
(available in Python 3.6 and later) are commonly used for this purpose, allowing for more readable and maintainable code.
```
# Lists and Tuples
``` bash
Lists and tuples are fundamental data structures in Python that allow for the storage and manipulation of collections of items.
Lists are mutable, meaning that their contents can be changed, while tuples are immutable, meaning once they are created, they
cannot be altered. Understanding the differences and use cases for each is crucial for effective programming in Python. 

Lists are typically used when you need a collection of items that may need to be modified, such as adding, removing, or changing
elements. They support various operations like appending, extending, and removing items. For example, you might use a list to keep
track of user inputs, where the data changes frequently.

Tuples, on the other hand, are suitable for cases where a collection of items should not be changed after creation. They can be
used to represent fixed collections of related items, such as coordinates or RGB color values. Since tuples are immutable,
they can also be used as keys in dictionaries, which is not possible with lists.
```
# Dictionaries
``` bash
Dictionaries are essential data structures in Python that store key-value pairs. They provide efficient retrieval and
organization of data, making them a cornerstone for various programming tasks. Their ability to allow constant time
complexity on average for lookups, insertions, and deletions makes them particularly useful for applications that
require fast data access. Additionally, dictionaries can accommodate a wide range of data types for both keys and values,
giving programmers flexibility in how they structure their data.

Creating and Using Dictionaries

Indeed, dictionaries in Python are versatile data structures that allow for the storage of key-value pairs. You can create
them using curly braces, like this: {key1: value1, key2: value2}, or by using the dict() function, such as dict(key1=value1,
key2=value2). Accessing a value is straightforward; you simply reference the key within square brackets, like dictionary[key].
This structure makes dictionaries very useful for organizing and retrieving data efficiently.

Iterating Through Dictionaries

Python provides several methods to iterate over dictionaries. You can use a for loop to go through the keys directly by iterating
over the dictionary itself. To access values, you can use the values() method, and to retrieve both keys and values together,
you can use the items() method. This flexibility allows for various ways to access and manipulate dictionary content efficiently.

Handling Missing Keys

The .get() method in a dictionary is a useful way to handle cases where a key may not be present. By allowing you to specify a
default value, it prevents errors that would occur if you tried to access a key that is not available. This means you can write
more robust code, as it handles missing keys gracefully without raising exceptions. For example, using my_dict.get('key',
'default_value') would return 'default_value' if 'key' does not exist in my_dict, thus ensuring smooth execution of your program.

Important Use Cases for Dictionaries

Dictionaries play a crucial role in programming and data management. They allow for fast lookups and provide a clear way to
associate keys with values. This makes them ideal for tasks like counting occurrences of elements in a dataset, where each unique
element can be a key and its count the associated value. Similarly, dictionaries can be used to group data by categorizing items
based on specific keys, enabling efficient data retrieval and analysis. Additionally, they are useful for mapping relationships,
as they can represent complex associations between different entities, such as objects and their attributes, or users and their
preferences. Overall, dictionaries enhance the organization and accessibility of information across various applications.
```
# Decision making statements
``` bash
Indeed, in Python, decision-making statements such as if, elif, and else are fundamental for controlling the flow of a program.
They allow the programmer to specify different blocks of code to be executed depending on whether certain conditions are true or false.
This branching logic is essential for creating dynamic and responsive applications, as it enables the program to make choices based on
user input, variable values, or other criteria. Here's a basic example of how these statements can be used:

if condition1:
    # execute this block if condition1 is true
elif condition2:
    # execute this block if condition1 is false and condition2 is true
else:
    # execute this block if both condition1 and condition2 are false

This structure facilitates complex decision-making processes within your code.

The 'if' Statement

The 'if' statement in Python evaluates a condition and executes the block of code that follows if the condition
is true. It is fundamental to controlling the flow of a program. Key aspects include using conditions with comparison
operators, logical operators, and allowing for the inclusion of 'elif' and 'else' statements to handle multiple cases
or alternative scenarios. This makes the control flow more versatile and enables developers to create dynamic and
responsive applications.

The 'elif' Statement

The 'elif' statement is an essential part of control flow in programming. It stands for "else if" and allows for checking
additional conditions after an initial 'if' statement. When using 'elif', you can define multiple potential conditions,
and the program will evaluate them in order. As soon as one of the conditions evaluates to true, the corresponding block
of code will execute, and the rest of the conditions will be ignored. This makes 'elif' especially useful for creating more
complex decision-making structures, where multiple pathways or outcomes are possible based on different criteria.

The 'else' Statement

That is correct. The 'else' statement serves as a final fallback in a conditional structure, allowing for a default action
to be taken when all previous 'if' and 'elif' conditions evaluate to false. This ensures that every possible outcome is
addressed in the program's logic.
```
# Looping Statements
``` bash
For loops are used to iterate over a sequence, such as a list, tuple, or string. The syntax typically involves defining a loop
variable that takes on the value of each item in the sequence one at a time. This allows developers to execute a block of code
for each element, making it easy to manipulate or analyze data.

While loops, on the other hand, continue to execute as long as a specified condition remains true. This type of loop is useful
when the number of iterations is not known in advance. Developers must be careful to ensure that the condition will eventually
evaluate to false, to avoid creating an infinite loop.

Both loop types can be controlled with the break and continue statements. The break statement exits the loop entirely, while the
continue statement skips the current iteration and proceeds to the next one.

How For Loops Work

For loops in Python are indeed a powerful feature that allows for the easy iteration over various collections. The basic syntax
is as follows: you start with the 'for' keyword, followed by a variable name that will take the value of each item in the collection,
followed by the 'in' keyword and the collection itself, and finally, you end with a colon. After this, you can write a block of code
that will execute for each item in the collection. The indentation of the block is crucial as it indicates which statements belong
to the loop. This structure makes it simple to perform operations on each element, making code cleaner and more readable.

Understanding While Loops

While loops continue to execute a block of code as long as a given condition is true. This type of loop is particularly useful when
the number of iterations is not predetermined. The syntax includes the 'while' keyword, a condition, and a block of code that gets
executed as long as the condition holds. In practical use, the loop often modifies variables within its body to ensure that the condition
will eventually become false, preventing an infinite loop.

Using Control Statements in Loops

Control statements like 'break' and 'continue' are essential for managing loop execution in programming. The 'break' statement
is used to terminate the loop entirely, causing the program to exit the loop and continue executing the code that follows it.
In contrast, the 'continue' statement skips the current iteration of the loop and moves directly to the next iteration, which means
that the remaining code within the loop's block for that iteration is not executed. These statements provide greater control over loop
behavior and help to implement conditional logic effectively.
```
# Functions 
``` bash
Functions are essential building blocks that allow developers to create modular and reusable code. A function is defined using the
'def' keyword followed by the function name, parentheses for parameters, and a colon. They can take input arguments and may return a value.
Understanding functions enhances code readability and maintainability.

Definition: In Python, a function is defined using the 'def' keyword followed by the function name and parentheses. Inside the parentheses,
you can include parameters that allow you to pass data into the function. The function body contains the code that runs when the function is called.
Return statement: Functions in Python can return values using the 'return' statement. Once a return statement is executed, the function exits, and
the value is sent back to the caller. If there is no return statement, the function will return 'None' by default.
Parameter types: Python supports various types of parameters, including positional, keyword, arbitrary positional (*args), and arbitrary keyword
(**kwargs) parameters. This flexibility allows for functions to handle a varying number of arguments, making them more versatile.
Default parameter values: Functions can have default values for parameters, which are used if no argument is provided during the function call.
This feature simplifies function calls and enhances usability, allowing for optional parameters.
Scope of varibles: Variables defined inside a function are local and cannot be accessed outside of that function. This encapsulation helps
avoid naming conflicts and keeps the global namespace clean.
```
# Modules

``` bash
A Python module is a file containing Python code (functions, variables, classes, or statements) that you can reuse in other Python programs by importing it.
Modules help in organizing code and reusing functionality.

Any .py file can be a module.

Step 1: Create a module mymodule.py
# mymodule.py
def greet(name):
    return f"Hello, {name}!"

Step 2: Use the module in another Python file
# main.py
import mymodule

print(mymodule.greet("Alice"))

Output:

Hello, Alice!
```
# Introduction to File Handling
``` bash
File handling is a crucial aspect of programming that allows developers to read from and write to files on the system. It enables
the storage and retrieval of data, making it an essential component for applications that require data persistence.

The basic operations associated with file handling in Python include:

1. Opening a file: This operation involves specifying the file's name and the mode in which it should be opened, such as read,
write, or append.

2. Reading from a file: Once a file is opened in read mode, data can be read using various methods, such as reading the entire content,
reading line by line, or reading a specific number of bytes.

3. Writing to a file: In write mode, data can be written to the file. This can either overwrite existing content or add new content
depending on the mode used.

4. Closing a file: After operations are complete, it's important to close the file to free up system resources and ensure that
all data is saved properly.

These operations are typically performed using Python's built-in functions which provides a convenient way to handle file resources efficiently.

Reading and Writing Text Files
Reading and writing text files is a straightforward process, typically accomplished using the built-in open function, which returns a file object. This file object provides methods for reading and writing operations.

To read a text file, you can use the following basic syntax:

1. Open the file in read mode ('r'):
file = open('filename.txt', 'r')

2. Read the contents using one of the methods like read, readline, or readlines:
content = file.read()  # Reads the entire file
line = file.readline()  # Reads one line from the file
lines = file.readlines()  # Reads all lines into a list

3. Always close the file after you're done to free up resources:
file.close()
A more common practice is to use a `with` statement, which automatically handles file closing:

with open('filename.txt', 'r') as file:
    content = file.read()


To write to a text file, you can open the file in write mode ('w') or append mode ('a'):

1. Open the file in write mode ('w'):
file = open('filename.txt', 'w')

2. Write content to the file using the write or writelines methods:
file.write('Hello, World!\n')  # Writes a single line
file.writelines(['Line 1\n', 'Line 2\n'])  # Writes multiple lines

3. Close the file:
file.close()
```
# Lab Manual
``` bash
1. Python program to check whether the given number is even
or not.
number = input("Enter a number ")
x = int(number)%2
if x == 0:
print(" The number is Even ")
else:
print(" The number is odd ")

2. Python program to convert the temperature in degree
centigrade to Fahrenheit
c = input(" Enter temperature in Centigrade: ")
f = (9*(int(c))/5)+32
print(" Temperature in Fahrenheit is: ", f)

3. Python program to find the area of a triangle whose sides
are given
import math
a = float(input("Enter the length of side a: "))
b = float(input("Enter the length of side b: "))
c = float(input("Enter the length of side c: "))
s = (a+b+c)/2
area = math.sqrt(s*(s-a)*(s-b)*(s-c))
print(" Area of the triangle is: ", area)

4. Python program to find out the average of a set of integers
count = int(input("Enter the count of numbers: "))
i = 0
sum = 0
for i in range(count):
x = int(input("Enter an integer: "))
sum = sum + x
avg = sum/count
print(" The average is: ", avg)

5. Python program to find the product of a set of real
numbers
i = 0
product = 1
count = int(input("Enter the number of real numbers: "))
for i in range(count):
x = float(input("Enter a real number: "))
product = product * x
print("The product of the numbers is: ", product)

6. Python program to find the circumference and area of a
circle with a given radius
import math
r = float(input("Input the radius of the circle: "))
c = 2 * math.pi * r
area = math.pi * r * r
print("The circumference of the circle is: ", c)
print("The area of the circle is: ", area)

7. Python program to check whether the given integer is a
multiple of 5
number = int(input("Enter an integer: "))
if(number%5==0):
print(number, "is a multile of 5")
else:
print(number, "is not a multiple of 5")

8. Python program to check whether the given integer is a
multiple of both 5 and 7
number = int(input("Enter an integer: "))
if((number%5==0)and(number%7==0)):
print(number, "is a multiple of both 5 and 7")
else:
print(number, "is not a multiple of both 5 and 7")

9. Python program to find the average of 10 numbers using
while loop
count = 0
sum = 0.0
while(count<10):
number = float(input("Enter a real number: "))
count=count+1
sum = sum+number
avg = sum/10;
print("Average is :",avg)

10. Python program to display the given integer in reverse
manner
number = int(input("Enter a positive integer: "))
rev = 0
while(number!=0):
digit = number%10
rev = (rev*10)+digit
number = number//10
print(rev)

11. Python program to find the geometric mean of n numbers
c = 0
p = 1.0
count = int(input("Enter the number of values: "))
while(c<count):
x = float(input("Enter a real number: "))
c = c+1
p = p * x
gm = pow(p,1.0/count)
print("The geometric mean is: ",gm

12. Python program to find the sum of the digits of an integer
using while loop
sum = 0
number = int(input("Enter an integer: "))
while(number!=0):
digit = number%10
sum = sum+digit
number = number//10
print("Sum of digits is: ", sum)

13. Python program to display all the multiples of 3 within
the range 10 to 50
for i in range(10,50):
if (i%3==0):
print(i)

14. Python program to display all integers within the range
100-200 whose sum of digits is an even number
for i in range(100,200):
num = i
sum = 0
while(num!=0):
digit = num%10
sum = sum + digit
num = num//10
if(sum%2==0):
print(i)

15. Python program to check whether the given integer is a
prime number or not
num = int(input("Enter an integer greater than 1: "))
isprime = 1 #assuming that num is prime
for i in range(2,num//2):
if (num%i==0):
isprime = 0
break
if(isprime==1):
print(num, "is a prime number")
else:
print(num, "is not a prime number")

16. Python program to generate the prime numbers from 1 to N
num =int(input("Enter the range: "))
for n in range(2,num):
for i in range(2,n):
if(n%i==0):
break
else:
print(n)

17. Python program to find the roots of a quadratic equation
import math
a = float(input("Enter the first coefficient: "))
b = float(input("Enter the second coefficient: "))
c = float(input("Enter the third coefficient: "))
if (a!=0.0):
d = (b*b)-(4*a*c)
if (d==0.0):
print("The roots are real and equal.")
r = -b/(2*a)
print("The roots are ", r,"and", r)
elif(d>0.0):
print("The roots are real and distinct.")
r1 = (-b+(math.sqrt(d)))/(2*a)
r2 = (-b-(math.sqrt(d)))/(2*a)
print("The root1 is: ", r1)
print("The root2 is: ", r2)
else
print("The roots are imaginary.")
rp = -b/(2*a)
ip = math.sqrt(-d)/(2*a)
print("The root1 is: ", rp, "+ i",ip)
print("The root2 is: ", rp, "- i",ip)
else:
print("Not a quadratic equation.")

18. Python program to print the numbers from a given
number n till 0 using recursion
def print_till_zero(n):
if (n==0):
return
print(n)
n=n-1
print_till_zero(n)
print_till_zero(8)

19. Python program to find the factorial of a number using
recursion
def fact(n):
if n==1:
f=1
else:
f = n * fact(n-1)
return f
num = int(input("Enter an integer: "))
result = fact(num)
print("The factorial of", num, " is: ", result)

20. Python program to display the sum of n numbers using a
list
numbers = []
num = int(input('How many numbers: '))
for n in range(num)
x = int(input('Enter number '))
numbers.append(x)
print("Sum of numbers in the given list is :", sum(numbers))

21. Python program to implement linear search
numbers = [4,2,7,1,8,3,6]
f = 0 #flag
x = int(input("Enter the number to be found out: "))
for i in range(len(numbers)):
if (x==numbers[i]):
print(" Successful search, the element is found at position", i)
f = 1
break
if(f==0):
print("Oops! Search unsuccessful")

22. Python program to implement binary search
def binarySearch(numbers, low, high, x):
if (high >= low):
mid = low + (high - low)//2
if (numbers[mid] == x):
return mid
elif (numbers[mid] > x):
return binarySearch(numbers, low, mid-1, x)
else:
return binarySearch(numbers, mid+1, high, x)
else:
return -1
numbers = [ 1,4,6,7,12,17,25 ] #binary search requires sorted numbers
x = 7
result = binarySearch(numbers, 0, len(numbers)-1, x)
if (result != -1):
print("Search successful, element found at position ", result)
else:
print("The given element is not present in the array")

23. Python program to find the odd numbers in an array
numbers = [8,3,1,6,2,4,5,9]
count = 0
for i in range(len(numbers)):
if(numbers[i]%2!=0):
count = count+1
print("The number of odd numbers in the list are: ", count)

24. Python program to find the largest number in a list
without using built-in functions
numbers = [3,8,1,7,2,9,5,4]
big = numbers[0]
position = 0
for i in range(len(numbers)):
if (numbers[i]>big):
big = numbers[i]
position = i
print("The largest element is ",big," which is found at position
",position)

25. Python program to insert a number to any position in a list
numbers = [3,4,1,9,6,2,8]
print(numbers)
x = int(input("Enter the number to be inserted: "))
y = int(input("Enter the position: "))
numbers.insert(y,x)
print(numbers)
26. Python program to delete an element from a list by index
numbers = [3,4,1,9,6,2,8]
print(numbers)
x = int(input("Enter the position of the element to be deleted: "))
numbers.pop(x
print(numbers)

27. Python program to check whether a string is palindrome
or not
def rev(inputString):
return inputString[::-1]
def isPalindrome(inputString):
reverseString = rev(inputString)
if (inputString == reverseString):
return True
return False
s = input("Enter a string: ")
result = isPalindrome(s)
if result == 1:
print("The string is palindrome")
else:
print("The string is not palindrome")

28. Python program to implement matrix addition
X = [[8,5,1],
[9 ,3,2],
[4 ,6,3]]
Y = [[8,5,3],
[9,5,7],
[9,4,1]]
result = [[0,0,0],
[0,0,0],
[0,0,0]]
for i in range(len(X)):
for j in range(len(X[0])):
result[i][j] = X[i][j] + Y[i][j]
for k in result:
print(k)

29. Python program to implement matrix multiplication
X = [[8,5,1],
[9 ,3,2],
[4 ,6,3]]
Y = [[8,5,3],
[9,5,7],
[9,4,1]]
result = [[0,0,0,0],
[0,0,0,0],
[0,0,0,0]]
for i in range(len(X)):
for j in range(len(Y[0])):
for k in range(len(Y)):
result[i][j] += X[i][k] * Y[k][j]
for x in result:
print(x)

30. Python program to check leap year
year = int(input("Enter a year: "))
if (year % 4) == 0:
if (year % 100) == 0:
if (year % 400) == 0:
print(year, " is a leap year")
else:
print(year, " is not a leap year")
else:
print(year, " is a leap year")
else:
print(year, " is not a leap year")

31. Python program to find the Nth term in a Fibonacci series
using recursion
def Fib(n):
if n<0:
print("The input is incorrect.")
elif n==1:
return 0
elif n==2:
return 1
else:
return Fib(n-1)+Fib(n-2)
print(Fib(7))

32. Python program to print Fibonacci series using iteration
a = 0
b = 1
n=int(input("Enter the number of terms in the sequence: "))
print(a,b,end=" ")
while(n-2):
c=a+b
a,b = b,c
print(c,end=" ")
n=n-1

33. Python program to print all the items in a dictionary
phone_book = {
'John' : [ '8592970000', 'john@xyzmail.com' ],
'Bob': [ '7994880000', 'bob@xyzmail.com' ],
'Tom' : [ '9749552647' , 'tom@xyzmail.com' ]
}
for k,v in phone_book.items():
print(k, ":", v)

34. Python program to implement a calculator to do basic
operations
def add(x,y):
print(x+y)
def subtract(x,y):
print(x-y)
def multiply(x,y):
print(x*y)
def divide(x,y):
print(x/y)
print("Enter two numbers")
n1=input()
n2=input()
print("Enter the operation +,-,,/ ")
op=input()
if op=='+':
add(int(n1),int(n2))
elif op=='-':
subtract(int(n1),int(n2))
elif op=='':
multiply(int(n1),int(n2))
elif op=='/':
divide(int(n1),int(n2))
else:
print(" Invalid entry ")
```

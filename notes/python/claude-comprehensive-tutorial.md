# 150 Advanced Python Questions & Answers - Complete 2026 Mastery Guide

## Table of Contents
1. Core Fundamentals & Data Structures (Q1-20)
2. Object-Oriented Programming & Design Patterns (Q21-35)
3. Functional Programming & Advanced Functions (Q36-50)
4. Iterators, Generators & Memory Optimization (Q51-65)
5. Decorators, Metaclasses & Descriptors (Q66-80)
6. Concurrency, Parallelism & Async Programming (Q81-95)
7. Performance Optimization & Profiling (Q96-110)
8. Testing, Debugging & Code Quality (Q111-120)
9. Modern Python Features & Best Practices (Q121-135)
10. Real-World Applications & Industry Standards (Q136-150)

---

## Section 1: Core Fundamentals & Data Structures (Q1-20)

### Q1: What are the key differences between Python's mutable and immutable types, and why does it matter?

**Answer:**
Mutable types can be changed after creation (lists, dicts, sets), while immutable types cannot (strings, tuples, integers, frozensets).

```python
# Mutable - list can be modified
my_list = [1, 2, 3]
my_list[0] = 100  # Works fine
print(my_list)  # [100, 2, 3]

# Immutable - tuple cannot be modified
my_tuple = (1, 2, 3)
# my_tuple[0] = 100  # TypeError!

# Why it matters: dictionary keys must be immutable
valid_dict = {(1, 2): "tuple key"}  # Valid
# invalid_dict = {[1, 2]: "list key"}  # TypeError!

# Memory implications
x = [1, 2, 3]
y = x  # Both point to same object
y.append(4)
print(x)  # [1, 2, 3, 4] - x is also modified!

# Immutable creates new objects
a = "hello"
b = a
b = b + " world"  # Creates new string
print(a)  # "hello" - unchanged
```

**Why it matters:** Affects function arguments, dictionary keys, performance, and prevents unexpected side effects.

---

### Q2: How does Python's memory management and garbage collection work?

**Answer:**
Python uses reference counting combined with a cycle-detecting garbage collector.

```python
import sys
import gc

# Reference counting
x = [1, 2, 3]
print(sys.getrefcount(x))  # Shows reference count

y = x  # Increases ref count
print(sys.getrefcount(x))

del y  # Decreases ref count
print(sys.getrefcount(x))

# Circular references - where cycle detector helps
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

node1 = Node(1)
node2 = Node(2)
node1.next = node2
node2.next = node1  # Circular reference!

# Manual garbage collection
gc.collect()  # Forces collection of circular references

# Check garbage collection stats
print(gc.get_count())  # Shows collection counts per generation

# Disable/enable garbage collection (rare cases)
gc.disable()
# ... performance-critical code ...
gc.enable()
```

**Key points:** Objects are deleted when ref count reaches 0. Generational GC handles circular references efficiently.

---

### Q3: What is the difference between `is` and `==`, and when should you use each?

**Answer:**
`is` checks object identity (same memory location), `==` checks value equality.

```python
# Integer caching (Python caches small integers)
a = 256
b = 256
print(a is b)  # True - same cached object

a = 257
b = 257
print(a is b)  # False in some contexts - different objects

# String interning
s1 = "hello"
s2 = "hello"
print(s1 is s2)  # True - strings are interned

# Lists - never use 'is' for value comparison
list1 = [1, 2, 3]
list2 = [1, 2, 3]
print(list1 == list2)  # True - same values
print(list1 is list2)  # False - different objects

# Best practice: use 'is' only for None, True, False
value = None
if value is None:  # Correct
    print("Value is None")

if value == None:  # Works but not idiomatic
    print("Value is None")

# Identity check for singleton pattern
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True - same instance
```

---

### Q4: How do Python's data structures (list, tuple, set, dict) differ in performance and use cases?

**Answer:**
Each structure is optimized for different operations:

```python
import timeit

# LIST - ordered, mutable, allows duplicates
# Best for: ordered collections, frequent indexing
my_list = [1, 2, 3, 4, 5]
my_list.append(6)  # O(1) average
my_list.insert(0, 0)  # O(n) - shifts all elements
print(3 in my_list)  # O(n) - linear search

# TUPLE - ordered, immutable, allows duplicates
# Best for: fixed collections, dictionary keys, memory efficiency
my_tuple = (1, 2, 3, 4, 5)
# my_tuple.append(6)  # Error - immutable
print(3 in my_tuple)  # O(n) - linear search
# 20-30% less memory than lists

# SET - unordered, mutable, no duplicates
# Best for: membership testing, removing duplicates, set operations
my_set = {1, 2, 3, 4, 5}
my_set.add(6)  # O(1) average
print(3 in my_set)  # O(1) - hash lookup!

# Remove duplicates efficiently
duplicates = [1, 2, 2, 3, 3, 3, 4]
unique = list(set(duplicates))

# DICT - key-value pairs, unordered (ordered in Python 3.7+)
# Best for: fast lookups, counting, caching
my_dict = {"a": 1, "b": 2, "c": 3}
print(my_dict["a"])  # O(1) - hash lookup
my_dict["d"] = 4  # O(1) average

# Performance comparison
def test_list():
    return 5000 in list(range(10000))

def test_set():
    return 5000 in set(range(10000))

print("List lookup:", timeit.timeit(test_list, number=10000))
print("Set lookup:", timeit.timeit(test_set, number=10000))
# Set is dramatically faster for membership testing!

# Use case examples
# List: maintaining order of items
tasks = ["task1", "task2", "task3"]

# Tuple: function returns multiple values
def get_coordinates():
    return (10, 20)

# Set: tracking unique visitors
visitors = set()
visitors.add("user1")
visitors.add("user1")  # Duplicate ignored
print(len(visitors))  # 1

# Dict: configuration, caching
config = {
    "host": "localhost",
    "port": 8080,
    "debug": True
}
```

---

### Q5: What are dictionary comprehensions and when should you use them over loops?

**Answer:**
Dictionary comprehensions provide concise syntax for creating dictionaries and are generally faster than loops.

```python
# Basic dictionary comprehension
squares = {x: x**2 for x in range(10)}
print(squares)  # {0: 0, 1: 1, 2: 4, ...}

# With conditional
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}

# From two lists
keys = ['a', 'b', 'c']
values = [1, 2, 3]
my_dict = {k: v for k, v in zip(keys, values)}

# Transforming existing dictionary
prices = {"apple": 0.50, "banana": 0.30, "orange": 0.70}
increased_prices = {item: price * 1.1 for item, price in prices.items()}

# Filtering dictionary
filtered = {k: v for k, v in prices.items() if v > 0.40}

# Swapping keys and values
inverted = {v: k for k, v in my_dict.items()}

# Real-world example: counting occurrences
text = "hello world hello python world"
word_count = {word: text.split().count(word) for word in set(text.split())}

# Better way using Counter (more efficient)
from collections import Counter
word_count = Counter(text.split())

# Nested dictionary comprehension
matrix = {
    i: {j: i * j for j in range(3)}
    for i in range(3)
}
# {0: {0: 0, 1: 0, 2: 0}, 1: {0: 0, 1: 1, 2: 2}, 2: {0: 0, 1: 2, 2: 4}}

# When NOT to use comprehensions
# Too complex - hurts readability
# bad = {k: complex_function(k) if condition1(k) else other_function(k) 
#        for k in items if condition2(k) and condition3(k)}

# Better as traditional loop
result = {}
for k in items:
    if condition2(k) and condition3(k):
        if condition1(k):
            result[k] = complex_function(k)
        else:
            result[k] = other_function(k)
```

---

### Q6: How do you properly handle mutable default arguments in Python?

**Answer:**
Mutable defaults are created once at function definition, causing unexpected behavior. Use None as default instead.

```python
# WRONG - mutable default argument bug
def add_item_wrong(item, items=[]):
    items.append(item)
    return items

print(add_item_wrong(1))  # [1]
print(add_item_wrong(2))  # [1, 2] - Unexpected!
print(add_item_wrong(3))  # [1, 2, 3] - Same list!

# CORRECT - use None as default
def add_item_correct(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item_correct(1))  # [1]
print(add_item_correct(2))  # [2] - Fresh list each time
print(add_item_correct(3))  # [3]

# Real-world example: database connection pool
class Database:
    def __init__(self, connections=None):
        if connections is None:
            connections = []
        self.connections = connections
    
    def add_connection(self, conn):
        self.connections.append(conn)

# Why it happens
def show_default(x=[]):
    print(id(x))  # Same ID every call - same object!
    x.append(1)
    return x

show_default()
show_default()
show_default()  # All use same list object

# Alternative: using factory function
from typing import List

def process_items(items: List[int] = None) -> List[int]:
    items = items or []  # Common idiom
    return [x * 2 for x in items]

# Or for more complex defaults
from typing import Dict

def create_config(options: Dict[str, any] = None) -> Dict[str, any]:
    if options is None:
        options = {
            "timeout": 30,
            "retries": 3,
            "debug": False
        }
    return options
```

---

### Q7: What are Python's special methods (dunder methods) and how do you use them?

**Answer:**
Special methods let you define behavior for built-in operations on custom objects.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # String representation
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    # Arithmetic operations
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    # Comparison operations
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        return (self.x**2 + self.y**2) < (other.x**2 + other.y**2)
    
    # Container operations
    def __len__(self):
        return 2
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Index out of range")
    
    # Context manager
    def __enter__(self):
        print("Entering context")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting context")
        return False
    
    # Callable
    def __call__(self, scalar):
        return self * scalar

# Usage
v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(v1)  # Calls __str__: (1, 2)
print(repr(v1))  # Calls __repr__: Vector(1, 2)

v3 = v1 + v2  # Calls __add__
print(v3)  # (4, 6)

print(v1 == Vector(1, 2))  # Calls __eq__: True
print(v1 < v2)  # Calls __lt__: True

print(len(v1))  # Calls __len__: 2
print(v1[0])  # Calls __getitem__: 1

with v1:  # Calls __enter__ and __exit__
    print("Inside context")

result = v1(5)  # Calls __call__
print(result)  # (5, 10)

# Practical example: custom dictionary-like class
class CaseInsensitiveDict:
    def __init__(self):
        self._data = {}
    
    def __setitem__(self, key, value):
        self._data[key.lower()] = value
    
    def __getitem__(self, key):
        return self._data[key.lower()]
    
    def __contains__(self, key):
        return key.lower() in self._data
    
    def __iter__(self):
        return iter(self._data)

d = CaseInsensitiveDict()
d["Name"] = "John"
print(d["name"])  # "John" - case insensitive
print("NAME" in d)  # True
```

---

### Q8: How does Python's `*args` and `**kwargs` work, and when should you use them?

**Answer:**
`*args` collects positional arguments as a tuple, `**kwargs` collects keyword arguments as a dictionary.

```python
# Basic usage
def print_args(*args, **kwargs):
    print("Positional:", args)
    print("Keyword:", kwargs)

print_args(1, 2, 3, name="John", age=30)
# Positional: (1, 2, 3)
# Keyword: {'name': 'John', 'age': 30}

# Unpacking arguments
def add(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
print(add(*numbers))  # Unpacks list as arguments

params = {"a": 1, "b": 2, "c": 3}
print(add(**params))  # Unpacks dict as keyword arguments

# Combining with regular parameters
def greet(greeting, *names, **options):
    sep = options.get("sep", ", ")
    return f"{greeting} {sep.join(names)}!"

print(greet("Hello", "Alice", "Bob", sep=" and "))
# "Hello Alice and Bob!"

# Forwarding arguments to another function
def wrapper(*args, **kwargs):
    print("Before function call")
    result = original_function(*args, **kwargs)
    print("After function call")
    return result

def original_function(x, y, z=10):
    return x + y + z

# Decorator pattern
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper

@logger
def multiply(a, b):
    return a * b

multiply(3, 4)

# Enforcing keyword-only arguments (Python 3+)
def create_user(name, *, email, age):
    # email and age MUST be passed as keyword arguments
    return {"name": name, "email": email, "age": age}

# create_user("John", "john@example.com", 30)  # Error!
user = create_user("John", email="john@example.com", age=30)  # Correct

# Positional-only arguments (Python 3.8+)
def process(a, b, /, c, d, *, e, f):
    # a, b: positional-only
    # c, d: positional or keyword
    # e, f: keyword-only
    return a + b + c + d + e + f

# process(1, 2, 3, 4, 5, 6)  # Error - e, f must be keyword
result = process(1, 2, 3, 4, e=5, f=6)  # Correct

# Real-world example: flexible API client
class APIClient:
    def request(self, endpoint, method="GET", **params):
        url = f"https://api.example.com/{endpoint}"
        # params can include: headers, timeout, auth, etc.
        print(f"{method} {url}")
        for key, value in params.items():
            print(f"  {key}: {value}")

client = APIClient()
client.request("users", method="POST", 
               headers={"Authorization": "Bearer token"},
               timeout=30,
               data={"name": "John"})
```

---

### Q9: What is the difference between shallow and deep copy in Python?

**Answer:**
Shallow copy creates a new object but references nested objects, while deep copy recursively copies all nested objects.

```python
import copy

# Shallow copy
original = [[1, 2, 3], [4, 5, 6]]
shallow = copy.copy(original)  # or original.copy() or original[:]

shallow[0][0] = 999
print(original)  # [[999, 2, 3], [4, 5, 6]] - Modified!
print(shallow)   # [[999, 2, 3], [4, 5, 6]]

# Deep copy
original = [[1, 2, 3], [4, 5, 6]]
deep = copy.deepcopy(original)

deep[0][0] = 999
print(original)  # [[1, 2, 3], [4, 5, 6]] - Unchanged!
print(deep)      # [[999, 2, 3], [4, 5, 6]]

# Visualization of the difference
original = [1, 2, [3, 4]]

# Shallow copy - inner list is shared
shallow = original[:]
print(id(original))  # Different ID
print(id(shallow))
print(id(original[2]))  # Same ID - shared reference!
print(id(shallow[2]))

# Deep copy - everything is new
deep = copy.deepcopy(original)
print(id(deep[2]))  # Different ID - independent copy

# Dictionary example
original_dict = {
    "name": "John",
    "scores": [90, 85, 95],
    "metadata": {"created": "2024-01-01"}
}

shallow_dict = original_dict.copy()
shallow_dict["scores"].append(100)
print(original_dict["scores"])  # [90, 85, 95, 100] - Modified!

deep_dict = copy.deepcopy(original_dict)
deep_dict["scores"].append(80)
print(original_dict["scores"])  # Unchanged

# When to use each
# Shallow copy: when you only need to modify top-level items
user = {"name": "Alice", "age": 30}
user_copy = user.copy()
user_copy["name"] = "Bob"  # Safe - only top level modified

# Deep copy: when you have nested structures you want independent
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

tree = Node(1, [Node(2), Node(3)])
tree_copy = copy.deepcopy(tree)  # Complete independent copy

# Performance consideration
import timeit

large_list = [[i] * 100 for i in range(1000)]

shallow_time = timeit.timeit(
    lambda: copy.copy(large_list),
    number=10000
)

deep_time = timeit.timeit(
    lambda: copy.deepcopy(large_list),
    number=10000
)

print(f"Shallow: {shallow_time:.4f}s")
print(f"Deep: {deep_time:.4f}s")
# Deep copy is significantly slower!
```

---

### Q10: How do you work with JSON data in Python effectively?

**Answer:**
Python's `json` module provides encoding/decoding with options for custom serialization.

```python
import json
from datetime import datetime
from decimal import Decimal

# Basic encoding and decoding
data = {
    "name": "John",
    "age": 30,
    "languages": ["Python", "JavaScript"],
    "active": True
}

# To JSON string
json_string = json.dumps(data, indent=2)
print(json_string)

# From JSON string
parsed = json.loads(json_string)

# File operations
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

with open("data.json", "r") as f:
    loaded = json.load(f)

# Custom serialization for non-JSON types
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

complex_data = {
    "timestamp": datetime.now(),
    "amount": Decimal("99.99"),
    "tags": {"python", "json", "tutorial"}
}

json_str = json.dumps(complex_data, cls=CustomEncoder, indent=2)
print(json_str)

# Custom deserialization
def datetime_decoder(dct):
    for key, value in dct.items():
        if key == "timestamp":
            dct[key] = datetime.fromisoformat(value)
    return dct

decoded = json.loads(json_str, object_hook=datetime_decoder)

# Handling nested JSON
nested = {
    "user": {
        "profile": {
            "name": "Alice",
            "settings": {
                "theme": "dark",
                "notifications": True
            }
        }
    }
}

# Safe access with get()
theme = nested.get("user", {}).get("profile", {}).get("settings", {}).get("theme")

# Better: using a helper function
def get_nested(data, *keys, default=None):
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return default
    return data if data is not None else default

theme = get_nested(nested, "user", "profile", "settings", "theme")

# JSON Schema validation (requires jsonschema)
# pip install jsonschema
from jsonschema import validate, ValidationError

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number", "minimum": 0}
    },
    "required": ["name", "age"]
}

valid_data = {"name": "John", "age": 30}
try:
    validate(instance=valid_data, schema=schema)
    print("Valid!")
except ValidationError as e:
    print(f"Invalid: {e.message}")

# Streaming large JSON files
def stream_json_array(file_path):
    """Stream large JSON array without loading everything in memory"""
    with open(file_path, 'r') as f:
        f.read(1)  # Skip opening bracket
        buffer = ""
        for line in f:
            buffer += line
            if line.strip().endswith('},'):
                yield json.loads(buffer.rstrip(','))
                buffer = ""

# Pretty printing
print(json.dumps(data, indent=2, sort_keys=True))

# Compact printing (for APIs)
compact = json.dumps(data, separators=(',', ':'))
```

---

### Q11: What are Python's context managers and how do you create custom ones?

**Answer:**
Context managers handle setup and teardown logic automatically using `with` statement.

```python
# Built-in context manager - file handling
with open("file.txt", "w") as f:
    f.write("Hello")
# File automatically closed, even if exception occurs

# Multiple context managers
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    content = infile.read()
    outfile.write(content.upper())

# Custom context manager - class-based
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to {self.connection_string}")
        self.connection = f"Connection to {self.connection_string}"
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection")
        self.connection = None
        # Return False to propagate exceptions
        # Return True to suppress exceptions
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}")
        return False

with DatabaseConnection("localhost:5432") as conn:
    print(f"Using {conn}")
    # raise ValueError("Something went wrong")

# Custom context manager - function-based using contextlib
from contextlib import contextmanager

@contextmanager
def timer(label):
    import time
    start = time.time()
    try:
        yield  # Code block executes here
    finally:
        end = time.time()
        print(f"{label}: {end - start:.2f}s")

with timer("Processing"):
    import time
    time.sleep(1)
    print("Doing work")

# Reusable file handler with error logging
@contextmanager
def safe_file_handler(filename, mode='r'):
    f = None
    try:
        f = open(filename, mode)
        yield f
    except IOError as e:
        print(f"Error accessing file: {e}")
        yield None
    finally:
        if f:
            f.close()

with safe_file_handler("data.txt", "r") as f:
    if f:
        content = f.read()

# Lock context manager for thread safety
from contextlib import contextmanager
import threading

lock = threading.Lock()

@contextmanager
def synchronized(lock):
    lock.acquire()
    try:
        yield
    finally:
        lock.release()

# Usage
shared_resource = 0
with synchronized(lock):
    shared_resource += 1

# Temporary directory context manager
import tempfile
import shutil
import os

@contextmanager
def temporary_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

with temporary_directory() as temp_dir:
    # Create files in temp directory
    file_path = os.path.join(temp_dir, "temp_file.txt")
    with open(file_path, "w") as f:
        f.write("Temporary data")
# temp_dir automatically deleted

# Suppress exceptions context manager
from contextlib import suppress

# Instead of try-except
with suppress(FileNotFoundError):
    os.remove("nonexistent_file.txt")
# No error raised if file doesn't exist

# Chaining context managers
from contextlib import ExitStack

def process_files(filenames):
    with ExitStack() as stack:
        files = [stack.enter_context(open(fn)) for fn in filenames]
        # All files automatically closed when exiting
        for f in files:
            print(f.read())

# Real-world example: database transaction
@contextmanager
def transaction(connection):
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
        print("Transaction committed")
    except Exception as e:
        connection.rollback()
        print(f"Transaction rolled back: {e}")
        raise

# Usage:
# with transaction(db_connection) as cursor:
#     cursor.execute("INSERT INTO users VALUES (?)", ("John",))
```

---

### Q12: How does Python's slice notation work and what are advanced slicing techniques?

**Answer:**
Slicing uses `[start:stop:step]` syntax with powerful features for sequence manipulation.

```python
# Basic slicing
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(numbers[2:7])     # [2, 3, 4, 5, 6] - start to stop-1
print(numbers[:5])      # [0, 1, 2, 3, 4] - from beginning
print(numbers[5:])      # [5, 6, 7, 8, 9] - to end
print(numbers[:])       # Full copy of list

# Step parameter
print(numbers[::2])     # [0, 2, 4, 6, 8] - every 2nd element
print(numbers[1::2])    # [1, 3, 5, 7, 9] - odd indices
print(numbers[::3])     # [0, 3, 6, 9] - every 3rd element

# Negative indices
print(numbers[-3:])     # [7, 8, 9] - last 3 elements
print(numbers[:-3])     # [0, 1, 2, 3, 4, 5, 6] - except last 3
print(numbers[-5:-2])   # [5, 6, 7] - range from end

# Reverse sequence
print(numbers[::-1])    # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
print(numbers[::-2])    # [9, 7, 5, 3, 1] - reverse, every 2nd

# Slice assignment
numbers = [0, 1, 2, 3, 4, 5]
numbers[2:4] = [20, 30, 40]  # Replace with different length
print(numbers)  # [0, 1, 20, 30, 40, 4, 5]

numbers[1:4] = []  # Delete elements
print(numbers)  # [0, 40, 4, 5]

# Insert elements
numbers = [0, 1, 2, 3, 4]
numbers[2:2] = [10, 20]  # Insert at index 2
print(numbers)  # [0, 1, 10, 20, 2, 3, 4]

# Slice object for reusability
LAST_THREE = slice(-3, None)
EVEN_INDICES = slice(None, None, 2)

data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(data[LAST_THREE])   # [7, 8, 9]
print(data[EVEN_INDICES]) # [0, 2, 4, 6, 8]

# Named slices improve readability
record = "john:doe:30:engineer:boston"
fields = record.split(':')

# Instead of magic numbers:
# name = fields[0] + ' ' + fields[1]
# age = fields[2]
# job = fields[3]

# Use named slices:
FIRST_NAME = slice(0, 1)
LAST_NAME = slice(1, 2)
AGE = slice(2, 3)
JOB = slice(3, 4)

first_name = fields[FIRST_NAME][0]
last_name = fields[LAST_NAME][0]

# String slicing
text = "Python Programming"
print(text[0:6])        # "Python"
print(text[7:])         # "Programming"
print(text[::-1])       # "gnimmargorP nohtyP" - reverse

# Check if string is palindrome
word = "racecar"
print(word == word[::-1])  # True

# Extract file extension
filename = "document.pdf"
extension = filename[filename.rfind('.'):]
# Or better:
extension = filename.split('.')[-1]

# Multi-dimensional slicing (NumPy-style with lists)
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]

# Get all rows, columns 1-2
result = [row[1:3] for row in matrix]
print(result)  # [[2, 3], [6, 7], [10, 11]]

# Practical examples

# 1. Pagination
def paginate(items, page_size=10, page_number=1):
    start = (page_number - 1) * page_size
    end = start + page_size
    return items[start:end]

all_items = list(range(100))
page_1 = paginate(all_items, page_size=10, page_number=1)
print(page_1)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 2. Rotating a list
def rotate_left(lst, n):
    n = n % len(lst)  # Handle n > len(lst)
    return lst[n:] + lst[:n]

numbers = [1, 2, 3, 4, 5]
print(rotate_left(numbers, 2))  # [3, 4, 5, 1, 2]

# 3. Chunk data into batches
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

data = list(range(10))
for chunk in chunks(data, 3):
    print(chunk)
# [0, 1, 2]
# [3, 4, 5]
# [6, 7, 8]
# [9]

# 4. Window sliding
def sliding_window(lst, window_size):
    for i in range(len(lst) - window_size + 1):
        yield lst[i:i + window_size]

numbers = [1, 2, 3, 4, 5]
for window in sliding_window(numbers, 3):
    print(window)
# [1, 2, 3]
# [2, 3, 4]
# [3, 4, 5]
```

---

### Q13: What are list comprehensions and how do they compare to map/filter?

**Answer:**
List comprehensions provide concise syntax for creating lists and are often more readable than map/filter.

```python
# Basic list comprehension
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Equivalent using map
squares_map = list(map(lambda x: x**2, range(10)))

# With conditional (filter)
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# Equivalent using filter + map
even_squares_functional = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(10))))

# Multiple conditions
result = [x for x in range(20) if x % 2 == 0 if x % 3 == 0]
print(result)  # [0, 6, 12, 18]

# if-else in comprehension (ternary operator)
values = [x if x % 2 == 0 else -x for x in range(10)]
print(values)  # [0, -1, 2, -3, 4, -5, 6, -7, 8, -9]

# Nested loops
matrix = [[i * j for j in range(3)] for i in range(3)]
print(matrix)
# [[0, 0, 0], [0, 1, 2], [0, 2, 4]]

# Flatten nested list
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested for item in sublist]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Set comprehension (removes duplicates)
text = "hello world"
unique_chars = {char.lower() for char in text if char.isalpha()}
print(unique_chars)  # {'h', 'e', 'l', 'o', 'w', 'r', 'd'}

# Dictionary comprehension
word_lengths = {word: len(word) for word in ["hello", "world", "python"]}
print(word_lengths)  # {'hello': 5, 'world': 5, 'python': 6}

# Generator expression (memory efficient)
sum_squares = sum(x**2 for x in range(1000000))  # No list created!

# Real-world examples

# 1. Data transformation
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]

names = [user["name"] for user in users]
adults = [user for user in users if user["age"] >= 30]

# 2. CSV processing
csv_lines = ["name,age,city", "Alice,30,NYC", "Bob,25,LA"]
data = [line.split(',') for line in csv_lines[1:]]  # Skip header
print(data)  # [['Alice', '30', 'NYC'], ['Bob', '25', 'LA']]

# 3. File processing
# filenames = [f for f in os.listdir('.') if f.endswith('.py')]

# 4. String manipulation
sentence = "Hello World Python"
words_upper = [word.upper() for word in sentence.split()]
print(words_upper)  # ['HELLO', 'WORLD', 'PYTHON']

# Performance comparison
import timeit

# List comprehension
def list_comp():
    return [x**2 for x in range(1000)]

# Map
def map_func():
    return list(map(lambda x: x**2, range(1000)))

# Traditional loop
def for_loop():
    result = []
    for x in range(1000):
        result.append(x**2)
    return result

print("List comp:", timeit.timeit(list_comp, number=10000))
print("Map:", timeit.timeit(map_func, number=10000))
print("For loop:", timeit.timeit(for_loop, number=10000))
# List comprehension is typically fastest!

# When NOT to use comprehensions
# Too complex - hurts readability
# Bad:
# result = [[cell.upper() if cell.isalpha() else cell.lower() 
#           for cell in row if len(cell) > 3] 
#          for row in matrix if sum(len(c) for c in row) > 10]

# Better as traditional loop with clear logic

# Walrus operator in comprehensions (Python 3.8+)
data = [1, 2, 3, 4, 5]
# Calculate expensive operation once
result = [y for x in data if (y := x**2) > 10]
print(result)  # [16, 25]
```

---

### Q14: How do you handle exceptions properly in Python?

**Answer:**
Use specific exception types, proper exception hierarchy, and clean error handling patterns.

```python
# Basic exception handling
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")

# Multiple exception types
try:
    value = int(input("Enter a number: "))
    result = 100 / value
except ValueError:
    print("Invalid number format")
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"Unexpected error: {e}")

# else clause - runs if no exception
try:
    file = open("data.txt", "r")
except FileNotFoundError:
    print("File not found")
else:
    # Only runs if no exception
    content = file.read()
    file.close()
    print("File read successfully")

# finally clause - always runs
try:
    connection = create_connection()
    data = connection.fetch_data()
except ConnectionError:
    print("Connection failed")
finally:
    # Always runs, even if exception occurs
    if 'connection' in locals():
        connection.close()

# Complete pattern
def process_file(filename):
    file = None
    try:
        file = open(filename, 'r')
        data = file.read()
        return process_data(data)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except PermissionError:
        print(f"No permission to read {filename}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    else:
        print("File processed successfully")
    finally:
        if file:
            file.close()

# Custom exceptions
class ValidationError(Exception):
    """Raised when data validation fails"""
    pass

class InsufficientFundsError(Exception):
    """Raised when account has insufficient funds"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Insufficient funds: balance={balance}, required={amount}")

def withdraw(account, amount):
    if amount > account.balance:
        raise InsufficientFundsError(account.balance, amount)
    account.balance -= amount

# Using custom exceptions
class Account:
    def __init__(self, balance):
        self.balance = balance

account = Account(100)
try:
    withdraw(account, 150)
except InsufficientFundsError as e:
    print(f"Transaction failed: {e}")
    print(f"Current balance: ${e.balance}")
    print(f"Attempted withdrawal: ${e.amount}")

# Exception chaining
def load_config(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Config file {filename} not found") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {filename}") from e

# Catching multiple exceptions as one
try:
    # some code
    pass
except (TypeError, ValueError, KeyError) as e:
    print(f"Input error: {e}")

# Context manager for exception handling
from contextlib import suppress

# Ignore specific exceptions
with suppress(FileNotFoundError):
    os.remove("temp_file.txt")

# Re-raising exceptions
def validate_age(age):
    try:
        age = int(age)
        if age < 0:
            raise ValueError("Age cannot be negative")
        return age
    except ValueError:
        print("Validation failed")
        raise  # Re-raise the same exception

# Exception groups (Python 3.11+)
# try:
#     # code that might raise multiple exceptions
#     pass
# except* ValueError as eg:
#     for e in eg.exceptions:
#         print(f"ValueError: {e}")
# except* TypeError as eg:
#     for e in eg.exceptions:
#         print(f"TypeError: {e}")

# Best practices

# 1. Be specific with exceptions
# Bad:
try:
    result = risky_operation()
except Exception:  # Too broad!
    pass

# Good:
try:
    result = risky_operation()
except (ConnectionError, TimeoutError) as e:
    handle_network_error(e)

# 2. Don't use bare except
# Bad:
try:
    something()
except:  # Catches everything, including KeyboardInterrupt!
    pass

# Good:
try:
    something()
except Exception as e:  # Doesn't catch system exits
    log_error(e)

# 3. Create exception hierarchy
class DatabaseError(Exception):
    """Base class for database exceptions"""
    pass

class ConnectionError(DatabaseError):
    """Database connection failed"""
    pass

class QueryError(DatabaseError):
    """Query execution failed"""
    pass

# Catch specific or general
try:
    execute_query()
except QueryError:
    # Handle query issues
    pass
except DatabaseError:
    # Handle all other database issues
    pass

# 4. Logging exceptions
import logging

try:
    risky_operation()
except Exception:
    logging.exception("Operation failed")  # Logs full traceback

# 5. Type hints for exceptions (Python 3.11+)
def divide(a: int, b: int) -> float:
    """Divide two numbers.
    
    Raises:
        ZeroDivisionError: If b is zero
        TypeError: If arguments are not numbers
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
```

---

### Q15: What are Python's built-in functions you should know?

**Answer:**
Python has 69 built-in functions providing essential operations without imports.

```python
# Type conversions
print(int("42"))        # 42
print(float("3.14"))    # 3.14
print(str(123))         # "123"
print(bool(0))          # False
print(list("hello"))    # ['h', 'e', 'l', 'l', 'o']
print(tuple([1, 2, 3])) # (1, 2, 3)
print(set([1, 2, 2, 3])) # {1, 2, 3}

# Numeric operations
print(abs(-42))         # 42
print(pow(2, 10))       # 1024 (same as 2**10)
print(round(3.14159, 2)) # 3.14
print(divmod(17, 5))    # (3, 2) - quotient and remainder
print(sum([1, 2, 3, 4])) # 10
print(min([5, 2, 8, 1])) # 1
print(max([5, 2, 8, 1])) # 8

# Advanced min/max with key
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78}
]
best_student = max(students, key=lambda s: s["grade"])
print(best_student)  # {'name': 'Bob', 'grade': 92}

# Sequence operations
numbers = [1, 2, 3, 4, 5]
print(len(numbers))     # 5
print(reversed(numbers)) # reversed object (iterator)
print(list(reversed(numbers))) # [5, 4, 3, 2, 1]

# sorted() - creates new sorted list
unsorted = [3, 1, 4, 1, 5, 9, 2]
print(sorted(unsorted))  # [1, 1, 2, 3, 4, 5, 9]
print(sorted(unsorted, reverse=True))  # [9, 5, 4, 3, 2, 1, 1]

# Sort by custom key
words = ["banana", "pie", "Washington", "book"]
print(sorted(words, key=len))  # ['pie', 'book', 'banana', 'Washington']
print(sorted(words, key=str.lower))  # Case-insensitive sort

# enumerate - get index and value
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# 0: apple
# 1: banana
# 2: cherry

# Start enumerate from different number
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}: {fruit}")

# zip - combine multiple iterables
names = ["Alice", "Bob", "Charlie"]
ages = [30, 25, 35]
cities = ["NYC", "LA", "Chicago"]

for name, age, city in zip(names, ages, cities):
    print(f"{name} is {age} years old and lives in {city}")

# zip stops at shortest sequence
short = [1, 2]
long = [10, 20, 30, 40]
print(list(zip(short, long)))  # [(1, 10), (2, 20)]

# Create dictionary from two lists
keys = ["name", "age", "city"]
values = ["John", 30, "Boston"]
person = dict(zip(keys, values))
print(person)

# map - apply function to each element
numbers = [1, 2, 3, 4, 5]
squared = map(lambda x: x**2, numbers)
print(list(squared))  # [1, 4, 9, 16, 25]

# map with multiple sequences
a = [1, 2, 3]
b = [10, 20, 30]
result = map(lambda x, y: x + y, a, b)
print(list(result))  # [11, 22, 33]

# filter - keep elements that pass test
numbers = range(10)
evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens))  # [0, 2, 4, 6, 8]

# any - True if any element is truthy
print(any([False, False, True, False]))  # True
print(any([0, [], "", None]))  # False

# Check if any number is negative
numbers = [1, 5, -3, 7]
has_negative = any(n < 0 for n in numbers)
print(has_negative)  # True

# all - True if all elements are truthy
print(all([True, True, True]))  # True
print(all([True, False, True]))  # False

# Check if all numbers are positive
numbers = [1, 5, 3, 7]
all_positive = all(n > 0 for n in numbers)
print(all_positive)  # True

# range - generate sequence of numbers
print(list(range(5)))        # [0, 1, 2, 3, 4]
print(list(range(2, 10)))    # [2, 3, 4, 5, 6, 7, 8, 9]
print(list(range(0, 10, 2))) # [0, 2, 4, 6, 8]

# Introspection functions
class MyClass:
    def __init__(self):
        self.x = 10

obj = MyClass()

print(type(obj))           # <class '__main__.MyClass'>
print(isinstance(obj, MyClass))  # True
print(hasattr(obj, 'x'))   # True
print(getattr(obj, 'x'))   # 10
setattr(obj, 'y', 20)
print(obj.y)               # 20
print(dir(obj))            # List all attributes

# id - unique identifier of object
a = [1, 2, 3]
b = [1, 2, 3]
print(id(a))  # Different from id(b)
print(a is b)  # False

# callable - check if object can be called
def func():
    pass

print(callable(func))      # True
print(callable(42))        # False
print(callable(list))      # True

# input/output
name = input("Enter name: ")  # Read from stdin
print("Hello", name)          # Write to stdout

# format numbers
print(format(1234, ','))      # "1,234"
print(format(0.123, '.2%'))   # "12.30%"

# bin, oct, hex - number conversions
print(bin(10))   # "0b1010"
print(oct(10))   # "0o12"
print(hex(255))  # "0xff"

# chr, ord - character conversions
print(chr(65))   # "A"
print(ord('A'))  # 65

# Complex number operations
c = complex(3, 4)  # 3 + 4j
print(abs(c))      # 5.0 - magnitude

# iter and next - manual iteration
my_list = [1, 2, 3]
iterator = iter(my_list)
print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
# print(next(iterator))  # StopIteration error

# With default value
print(next(iterator, "Done"))  # "Done" instead of error

# eval - evaluate string as code (use carefully!)
result = eval("2 + 3 * 4")
print(result)  # 14

# exec - execute string as code
code = """
def greet(name):
    return f"Hello, {name}"
    
print(greet("World"))
"""
exec(code)

# compile - compile code for later execution
code = compile("print('Hello')", "<string>", "exec")
exec(code)

# vars() - return __dict__ of object
print(vars(obj))  # {'x': 10, 'y': 20}

# globals() and locals()
def func():
    local_var = 42
    print(locals())  # Shows local variables
    print(globals())  # Shows global variables

# help - interactive documentation
help(sorted)  # Shows documentation for sorted()
```

---

### Q16: What are Python's string formatting methods and when to use each?

**Answer:**
Python offers %-formatting, str.format(), and f-strings (preferred in modern code).

```python
# 1. f-strings (Python 3.6+) - PREFERRED METHOD
name = "Alice"
age = 30
city = "NYC"

# Basic f-string
message = f"Hello, {name}!"
print(message)  # "Hello, Alice!"

# Expressions in f-strings
print(f"{name} will be {age + 1} next year")

# Formatting numbers
price = 19.99
print(f"Price: ${price:.2f}")  # "Price: $19.99"

# Alignment and padding
print(f"{name:>10}")   # "     Alice" - right aligned
print(f"{name:<10}")   # "Alice     " - left aligned
print(f"{name:^10}")   # "  Alice   " - centered

# Numbers with separators
large_num = 1000000
print(f"{large_num:,}")  # "1,000,000"

# Percentage
ratio = 0.857
print(f"{ratio:.1%}")  # "85.7%"

# Binary, octal, hex
num = 42
print(f"{num:b}")  # "101010" - binary
print(f"{num:o}")  # "52" - octal
print(f"{num:x}")  # "2a" - hex
print(f"{num:X}")  # "2A" - hex uppercase

# Datetime formatting
from datetime import datetime
now = datetime.now()
print(f"{now:%Y-%m-%d %H:%M:%S}")  # "2024-01-15 14:30:45"

# Debug f-strings (Python 3.8+)
x = 42
y = 10
print(f"{x=}, {y=}")  # "x=42, y=10"
print(f"{x + y=}")    # "x + y=52"

# 2. str.format() - Still useful for templates
template = "Hello, {}! You are {} years old."
message = template.format(name, age)
print(message)

# Named placeholders
template = "Hello, {name}! You live in {city}."
message = template.format(name=name, city=city)

# Positional and keyword combined
template = "{0} is {1} years old and lives in {city}"
message = template.format(name, age, city=city)

# Format specifications
print("{:.2f}".format(3.14159))  # "3.14"
print("{:,}".format(1000000))    # "1,000,000"
print("{:>10}".format("right"))  # "     right"

# Reusing arguments
template = "{0} loves {1}. {0} really loves {1}!"
print(template.format("Alice", "Python"))

# Dictionary unpacking
person = {"name": "Bob", "age": 25}
print("{name} is {age} years old".format(**person))

# 3. % formatting (old style - avoid in new code)
print("Hello, %s!" % name)
print("%s is %d years old" % (name, age))
print("Price: $%.2f" % price)

# Multiple values
print("Name: %s, Age: %d, City: %s" % (name, age, city))

# Dictionary formatting
print("%(name)s is %(age)d" % {"name": name, "age": age})

# Real-world examples

# 1. Logging with timestamps
def log(message):
    timestamp = datetime.now()
    print(f"[{timestamp:%Y-%m-%d %H:%M:%S}] {message}")

log("Application started")

# 2. Table formatting
def print_table(data):
    # Data: list of dictionaries
    headers = data[0].keys()
    
    # Calculate column widths
    widths = {h: max(len(str(h)), max(len(str(row[h])) for row in data)) 
              for h in headers}
    
    # Print header
    header_line = " | ".join(f"{h:<{widths[h]}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in data:
        print(" | ".join(f"{str(row[h]):<{widths[h]}}" for h in headers))

users = [
    {"name": "Alice", "age": 30, "city": "NYC"},
    {"name": "Bob", "age": 25, "city": "LA"},
    {"name": "Charlie", "age": 35, "city": "Chicago"}
]
print_table(users)

# 3. Progress bar
def progress_bar(current, total, width=50):
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r[{bar}] {percent:.1%}", end="", flush=True)

# 4. File size formatting
def format_bytes(bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0

print(format_bytes(1234567))  # "1.18 MB"

# 5. SQL query building (be careful of SQL injection!)
# Safe way - use parameterized queries
query_template = "SELECT * FROM users WHERE age > {min_age} AND city = {city}"
# Don't use: query = query_template.format(min_age=25, city=user_input)
# Use parameterized queries with your DB library instead

# 6. API response formatting
def format_api_response(status, data, elapsed_time):
    return f"""
    Status: {status}
    Response Time: {elapsed_time:.3f}s
    Data: {data}
    """

# 7. Currency formatting
def format_currency(amount, currency="USD"):
    symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{amount:,.2f}"

print(format_currency(1234.56))  # "$1,234.56"

# 8. Multiline f-strings
report = f"""
Sales Report
============
Date: {datetime.now():%Y-%m-%d}
Total Sales: ${12345.67:,.2f}
Number of Transactions: {42}
Average: ${12345.67/42:,.2f}
"""
print(report)
```

---

### Q17: How do Python's comparison and identity operators work?

**Answer:**
Comparison operators compare values, while identity operators check if objects are the same instance.

```python
# Comparison operators: ==, !=, <, >, <=, >=
print(5 == 5)     # True
print(5 != 3)     # True
print(5 > 3)      # True
print(5 >= 5)     # True

# Identity operators: is, is not
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)     # True - same values
print(a is b)     # False - different objects
print(a is c)     # True - same object

# Check identity
print(id(a))      # Memory address
print(id(b))      # Different address
print(id(c))      # Same as a

# Integer caching (-5 to 256)
x = 256
y = 256
print(x is y)     # True - cached

x = 257
y = 257
print(x is y)     # May be False - not cached

# None comparison - always use 'is'
value = None
if value is None:  # Correct
    print("Value is None")

if value == None:  # Works but not idiomatic
    print("Value is None")

# Boolean comparison
flag = True
if flag is True:   # Don't do this
    pass

if flag:           # Do this instead
    pass

# String interning
s1 = "hello"
s2 = "hello"
print(s1 is s2)    # True - strings are interned

s1 = "hello world"
s2 = "hello world"
print(s1 is s2)    # May be False - not interned

# Chained comparisons
x = 5
print(1 < x < 10)  # True - cleaner than: 1 < x and x < 10
print(10 > x >= 5) # True

# Custom comparison in classes
class Version:
    def __init__(self, major, minor):
        self.major = major
        self.minor = minor
    
    def __eq__(self, other):
        return self.major == other.major and self.minor == other.minor
    
    def __lt__(self, other):
        if self.major != other.major:
            return self.major < other.major
        return self.minor < other.minor
    
    def __le__(self, other):
        return self == other or self < other
    
    # __gt__, __ge__, __ne__ are automatically derived

v1 = Version(1, 0)
v2 = Version(2, 0)
v3 = Version(1, 5)

print(v1 < v2)     # True
print(v1 < v3)     # True
print(v2 > v3)     # True

# Membership operators: in, not in
fruits = ["apple", "banana", "cherry"]
print("apple" in fruits)      # True
print("grape" not in fruits)  # True

# Works with strings
text = "hello world"
print("world" in text)         # True
print("xyz" not in text)       # True

# Works with dictionaries (checks keys)
person = {"name": "Alice", "age": 30}
print("name" in person)        # True
print("Alice" in person)       # False - checks keys only
print("Alice" in person.values())  # True

# Logical operators: and, or, not
print(True and True)   # True
print(True or False)   # True
print(not True)        # False

# Short-circuit evaluation
def expensive_check():
    print("Expensive check called")
    return True

# 'and' short-circuits if first is False
if False and expensive_check():
    pass  # expensive_check() never called

# 'or' short-circuits if first is True
if True or expensive_check():
    pass  # expensive_check() never called

# Truthy and falsy values
# Falsy: False, None, 0, 0.0, '', [], {}, set()
# Everything else is truthy

if []:
    print("Not printed")  # Empty list is falsy

if [1, 2, 3]:
    print("Printed")      # Non-empty list is truthy

# Using 'or' for default values
name = input("Enter name: ") or "Anonymous"
# If empty string, uses "Anonymous"

# Using 'and' for conditional execution
user_logged_in = True
premium_user = True
user_logged_in and premium_user and show_premium_content()

# Comparison of different types
# Python 3 doesn't allow comparing incompatible types
# print(5 < "hello")  # TypeError in Python 3

# But None can be compared
print(None == None)    # True
print(None is None)    # True (preferred)

# Custom ordering with key functions
names = ["alice", "Bob", "CHARLIE"]
print(sorted(names))              # Case-sensitive
print(sorted(names, key=str.lower))  # Case-insensitive

# Complex sorting
students = [
    {"name": "Alice", "grade": 85, "age": 20},
    {"name": "Bob", "grade": 92, "age": 19},
    {"name": "Charlie", "grade": 85, "age": 21}
]

# Sort by grade, then by age
sorted_students = sorted(students, key=lambda s: (s["grade"], s["age"]))

# Reverse sorting
sorted_desc = sorted(students, key=lambda s: s["grade"], reverse=True)

---

### Q18: What are Python decorators and how do you create them?

**Answer:**
Decorators modify or enhance functions/classes without changing their code, using the @decorator syntax.

```python
# Simple decorator
def simple_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@simple_decorator
def say_hello():
    print("Hello!")

say_hello()
# Before function
# Hello!
# After function

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
# Hello, Alice!
# Hello, Alice!
# Hello, Alice!

# Preserving function metadata
from functools import wraps

def my_decorator(func):
    @wraps(func)  # Preserves original function's name, docstring, etc.
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def example():
    """Example function"""
    pass

print(example.__name__)  # "example" (not "wrapper")
print(example.__doc__)   # "Example function"

# Timer decorator
import time

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

slow_function()

# Caching decorator
def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))  # Fast due to caching!

# Better: use built-in lru_cache
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n < 2:
        return n
    return fibonacci_cached(n-1) + fibonacci_cached(n-2)

# Retry decorator
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    print(f"Attempt {attempts} failed: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def unreliable_api_call():
    import random
    if random.random() < 0.7:
        raise ConnectionError("API unavailable")
    return "Success"

# Authentication decorator
def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        user = kwargs.get('user')
        if not user or not user.get('authenticated'):
            raise PermissionError("Authentication required")
        return func(*args, **kwargs)
    return wrapper

@require_auth
def access_resource(resource_id, user=None):
    return f"Accessing resource {resource_id}"

# access_resource(123)  # PermissionError
access_resource(123, user={'authenticated': True})  # Works

# Validation decorator
def validate_types(**expected_types):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate argument types
            for arg_name, expected_type in expected_types.items():
                if arg_name in kwargs:
                    if not isinstance(kwargs[arg_name], expected_type):
                        raise TypeError(
                            f"{arg_name} must be {expected_type.__name__}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(name=str, age=int)
def create_user(name, age):
    return {"name": name, "age": age}

user = create_user(name="Alice", age=30)  # Works
# user = create_user(name="Alice", age="30")  # TypeError

# Class decorator
def singleton(cls):
    instances = {}
    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper

@singleton
class Database:
    def __init__(self):
        print("Creating database connection")

db1 = Database()  # "Creating database connection"
db2 = Database()  # No message - returns same instance
print(db1 is db2)  # True

# Method decorators
class MyClass:
    @staticmethod
    def static_method():
        print("Static method")
    
    @classmethod
    def class_method(cls):
        print(f"Class method of {cls.__name__}")
    
    @property
    def read_only(self):
        return self._value
    
    @read_only.setter
    def read_only(self, value):
        self._value = value

# Stacking decorators (applied bottom-up)
@timer
@memoize
def complex_calculation(n):
    return sum(range(n))

# Equivalent to:
# complex_calculation = timer(memoize(complex_calculation))

# Debug decorator
def debug(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result!r}")
        return result
    return wrapper

@debug
def add(a, b):
    return a + b

add(5, 3)
# Calling add(5, 3)
# add returned 8

# Rate limiting decorator
from collections import deque
from time import time

def rate_limit(max_calls, time_window):
    calls = deque()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time()
            # Remove old calls outside time window
            while calls and calls[0] < now - time_window:
                calls.popleft()
            
            if len(calls) >= max_calls:
                raise Exception(f"Rate limit exceeded: {max_calls} calls per {time_window}s")
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=5, time_window=60)
def api_call():
    return "API response"

---

### Q19: What is the difference between `@staticmethod`, `@classmethod`, and instance methods?

**Answer:**
These three method types serve different purposes in object-oriented programming.

```python
class MyClass:
    class_variable = "I'm a class variable"
    
    def __init__(self, value):
        self.instance_variable = value
    
    # Instance method - has access to instance (self)
    def instance_method(self):
        return f"Instance method can access: {self.instance_variable}"
    
    # Class method - has access to class (cls), not instance
    @classmethod
    def class_method(cls):
        return f"Class method can access: {cls.class_variable}"
    
    # Static method - no access to class or instance
    @staticmethod
    def static_method():
        return "Static method is like a regular function"

# Instance method usage
obj = MyClass("instance value")
print(obj.instance_method())
# "Instance method can access: instance value"

# Class method usage - can be called on class or instance
print(MyClass.class_method())
# "Class method can access: I'm a class variable"
print(obj.class_method())  # Also works

# Static method usage
print(MyClass.static_method())
print(obj.static_method())  # Also works

# Real-world use cases

# 1. Alternative constructors with @classmethod
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    @classmethod
    def from_string(cls, date_string):
        year, month, day = map(int, date_string.split('-'))
        return cls(year, month, day)
    
    @classmethod
    def today(cls):
        import datetime
        today = datetime.date.today()
        return cls(today.year, today.month, today.day)
    
    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"

# Multiple ways to create Date
date1 = Date(2024, 1, 15)
date2 = Date.from_string("2024-01-15")
date3 = Date.today()

print(date1)
print(date2)
print(date3)

# 2. Factory pattern with @classmethod
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients
    
    @classmethod
    def margherita(cls):
        return cls(["mozzarella", "tomatoes", "basil"])
    
    @classmethod
    def pepperoni(cls):
        return cls(["mozzarella", "tomatoes", "pepperoni"])
    
    def __repr__(self):
        return f"Pizza with {', '.join(self.ingredients)}"

pizza1 = Pizza.margherita()
pizza2 = Pizza.pepperoni()
print(pizza1)
print(pizza2)

# 3. Utility functions with @staticmethod
class StringUtils:
    @staticmethod
    def reverse(string):
        return string[::-1]
    
    @staticmethod
    def is_palindrome(string):
        clean = string.lower().replace(" ", "")
        return clean == clean[::-1]
    
    @staticmethod
    def truncate(string, length):
        return string[:length] + "..." if len(string) > length else string

print(StringUtils.reverse("hello"))
print(StringUtils.is_palindrome("racecar"))
print(StringUtils.truncate("Long text here", 8))

# 4. Counting instances with @classmethod
class Employee:
    num_employees = 0
    
    def __init__(self, name):
        self.name = name
        Employee.num_employees += 1
    
    @classmethod
    def get_employee_count(cls):
        return cls.num_employees
    
    @classmethod
    def from_string(cls, emp_string):
        name = emp_string.split('-')[0]
        return cls(name)

emp1 = Employee("Alice")
emp2 = Employee("Bob")
emp3 = Employee.from_string("Charlie-Developer")

print(Employee.get_employee_count())  # 3

# 5. Inheritance behavior
class Animal:
    species_count = 0
    
    @classmethod
    def add_species(cls):
        cls.species_count += 1
        return cls.species_count
    
    @staticmethod
    def make_sound():
        return "Some generic sound"

class Dog(Animal):
    @staticmethod
    def make_sound():
        return "Woof!"

class Cat(Animal):
    @staticmethod
    def make_sound():
        return "Meow!"

# Class method uses the actual class
print(Dog.add_species())    # Modifies Dog.species_count
print(Cat.add_species())    # Modifies Cat.species_count
print(Animal.species_count)  # 0 - unchanged

# Static methods can be overridden
print(Animal.make_sound())  # "Some generic sound"
print(Dog.make_sound())     # "Woof!"
print(Cat.make_sound())     # "Meow!"

# 6. Validation with @staticmethod
class Validator:
    @staticmethod
    def validate_email(email):
        import re
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone):
        import re
        pattern = r'^\d{3}-\d{3}-\d{4}
        return bool(re.match(pattern, phone))

print(Validator.validate_email("user@example.com"))  # True
print(Validator.validate_phone("123-456-7890"))      # True

# When to use each:
# - Instance method: When you need access to instance data
# - Class method: Alternative constructors, factory methods, class-level operations
# - Static method: Utility functions related to the class but not needing class/instance data

---

### Q20: How does Python's `with` statement work and what are context managers?

**Answer:**
The `with` statement ensures proper resource cleanup using context managers that implement `__enter__` and `__exit__`.

```python
# Basic file handling without context manager (old way)
file = open("data.txt", "r")
try:
    content = file.read()
finally:
    file.close()  # Must remember to close

# With context manager (better)
with open("data.txt", "r") as file:
    content = file.read()
# File automatically closed, even if exception occurs

# Multiple context managers
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    content = infile.read()
    outfile.write(content.upper())

# Creating custom context manager - class-based
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing {self.filename}")
        if self.file:
            self.file.close()
        # Return False to propagate exceptions
        # Return True to suppress exceptions
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress exceptions

with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")

# Creating custom context manager - function-based
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    print(f"Opening {filename}")
    file = open(filename, mode)
    try:
        yield file  # Control returns to with block here
    finally:
        print(f"Closing {filename}")
        file.close()

with file_manager("test.txt", "r") as f:
    content = f.read()

# Timer context manager
@contextmanager
def timer(label):
    import time
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{label} took {end - start:.4f} seconds")

with timer("Data processing"):
    # Expensive operation
    sum(range(1000000))

# Database transaction context manager
@contextmanager
def transaction(connection):
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
        print("Transaction committed")
    except Exception as e:
        connection.rollback()
        print(f"Transaction rolled back: {e}")
        raise

# Usage:
# with transaction(db_connection) as cursor:
#     cursor.execute("INSERT INTO users VALUES (?)", ("John",))
#     cursor.execute("UPDATE accounts SET balance = balance - 100")

# Temporary directory context manager
import tempfile
import shutil

@contextmanager
def temporary_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

with temporary_directory() as tmpdir:
    # Work with temporary directory
    filepath = os.path.join(tmpdir, "temp_file.txt")
    with open(filepath, "w") as f:
        f.write("Temporary data")
# Directory and contents automatically deleted

# Changing directory context manager
import os

@contextmanager
def working_directory(path):
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)

with working_directory("/tmp"):
    # Work in /tmp directory
    print(os.getcwd())  # /tmp
print(os.getcwd())  # Back to original directory

# Suppressing exceptions
from contextlib import suppress

# Instead of try-except for specific exceptions
with suppress(FileNotFoundError):
    os.remove("nonexistent_file.txt")
# No error raised

with suppress(ValueError, TypeError):
    int("not a number")
# Suppresses both ValueError and TypeError

# Redirecting stdout
from contextlib import redirect_stdout
import io

f = io.StringIO()
with redirect_stdout(f):
    print("This goes to StringIO")
    print("Not to console")

output = f.getvalue()
print(output)  # Now prints to console

# Lock context manager for threading
import threading

lock = threading.Lock()

def thread_safe_operation():
    with lock:
        # Critical section - only one thread at a time
        shared_resource.modify()

# ExitStack for dynamic number of context managers
from contextlib import ExitStack

def process_files(filenames):
    with ExitStack() as stack:
        # Open all files
        files = [stack.enter_context(open(fn)) for fn in filenames]
        # Process files
        for f in files:
            process(f.read())
    # All files automatically closed

# Nested context managers
@contextmanager
def managed_resource(name):
    print(f"Acquiring {name}")
    try:
        yield name
    finally:
        print(f"Releasing {name}")

with managed_resource("Resource A"):
    with managed_resource("Resource B"):
        print("Using both resources")

# AsyncIO context managers (Python 3.7+)
import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring async resource")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing async resource")
        await asyncio.sleep(0.1)
        return False

async def use_async_resource():
    async with AsyncResource() as resource:
        print("Using async resource")

# asyncio.run(use_async_resource())

# Best practices:
# 1. Always use context managers for resources (files, connections, locks)
# 2. Prefer function-based (@contextmanager) for simple cases
# 3. Use class-based for complex cleanup logic or reusable managers
# 4. Context managers ensure cleanup even when exceptions occur

---

## Section 2: Object-Oriented Programming & Design Patterns (Q21-35)

### Q21: How do Python classes and inheritance work?

**Answer:**
Python supports single and multiple inheritance with Method Resolution Order (MRO) for handling conflicts.

```python
# Basic class
class Dog:
    # Class variable (shared by all instances)
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        # Instance variables (unique to each instance)
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says Woof!"
    
    def __str__(self):
        return f"{self.name} is {self.age} years old"

# Creating instances
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(dog1.bark())  # "Buddy says Woof!"
print(dog1.species)  # "Canis familiaris"
print(dog1)  # "Buddy is 3 years old"

# Single inheritance
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement speak()")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# Polymorphism
animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(animal.speak())

# Calling parent methods with super()
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    def info(self):
        return f"{self.brand} {self.model}"

class Car(Vehicle):
    def __init__(self, brand, model, doors):
        super().__init__(brand, model)  # Call parent constructor
        self.doors = doors
    
    def info(self):
        parent_info = super().info()  # Call parent method
        return f"{parent_info} with {self.doors} doors"

car = Car("Toyota", "Camry", 4)
print(car.info())  # "Toyota Camry with 4 doors"

# Multiple inheritance
class Flyer:
    def fly(self):
        return "Flying in the sky"

class Swimmer:
    def swim(self):
        return "Swimming in water"

class Duck(Animal, Flyer, Swimmer):
    def speak(self):
        return f"{self.name} says Quack!"

duck = Duck("Donald")
print(duck.speak())  # "Donald says Quack!"
print(duck.fly())    # "Flying in the sky"
print(duck.swim())   # "Swimming in water"

# Method Resolution Order (MRO)
print(Duck.__mro__)
# (<class 'Duck'>, <class 'Animal'>, <class 'Flyer'>, <class 'Swimmer'>, <class 'object'>)

# Diamond problem
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

d = D()
print(d.method())  # "B" - follows MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

# Property decorators
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2
    
    @property
    def circumference(self):
        return 2 * 3.14159 * self._radius

circle = Circle(5)
print(circle.radius)  # 5
print(circle.area)    # 78.53975
circle.radius = 10    # Uses setter
# circle.area = 100   # Error: can't set attribute

# Private and protected attributes
class BankAccount:
    def __init__(self, balance):
        self._balance = balance  # Protected (by convention)
        self.__pin = "1234"      # Name mangling (private)
    
    def get_balance(self):
        return self._balance
    
    def _internal_method(self):  # Protected method
        return "Internal use"
    
    def __verify_pin(self, pin):  # Private method
        return pin == self.__pin

account = BankAccount(1000)
print(account._balance)  # Works but discouraged
# print(account.__pin)   # AttributeError
print(account._BankAccount__pin)  # Name mangling - still accessible

# Class methods and static methods
class MathOperations:
    @staticmethod
    def add(a, b):
        return a + b
    
    @classmethod
    def multiply_by_two(cls, a):
        return cls.add(a, a)  # Can call other class methods

print(MathOperations.add(5, 3))  # 8
print(MathOperations.multiply_by_two(5))  # 10

# Abstract base classes
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# shape = Shape()  # TypeError: Can't instantiate abstract class
rect = Rectangle(5, 3)
print(rect.area())  # 15

---

### Q22: What are dataclasses and when should you use them?

**Answer:**
Dataclasses (Python 3.7+) reduce boilerplate for classes that primarily store data.

```python
from dataclasses import dataclass, field
from typing import List

# Without dataclass - lots of boilerplate
class PersonOld:
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email
    
    def __repr__(self):
        return f"Person(name={self.name}, age={self.age}, email={self.email})"
    
    def __eq__(self, other):
        if not isinstance(other, PersonOld):
            return False
        return (self.name, self.age, self.email) == (other.name, other.age, other.email)

# With dataclass - clean and concise
@dataclass
class Person:
    name: str
    age: int
    email: str

# Automatically generates __init__, __repr__, __eq__
person = Person("Alice", 30, "alice@example.com")
print(person)  # Person(name='Alice', age=30, email='alice@example.com')

person2 = Person("Alice", 30, "alice@example.com")
print(person == person2)  # True - __eq__ generated automatically

# Default values
@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0  # Default value
    tags: List[str] = field(default_factory=list)  # Mutable default

product = Product("Laptop", 999.99)
print(product)  # Product(name='Laptop', price=999.99, quantity=0, tags=[])

# Frozen dataclasses (immutable)
@dataclass(frozen=True)
class Point:
    x: int
    y: int

point = Point(10, 20)
# point.x = 30  # FrozenInstanceError

# Can be used as dictionary keys
points_dict = {Point(0, 0): "origin", Point(1, 1): "diagonal"}

# Order comparison
@dataclass(order=True)
class Student:
    name: str
    grade: float

students = [
    Student("Alice", 85.5),
    Student("Bob", 92.0),
    Student("Charlie", 78.5)
]
print(sorted(students))  # Sorted by all fields in order

# Custom ordering with sort_index
@dataclass(order=True)
class Task:
    priority: int = field(compare=True)
    name: str = field(compare=False)  # Don't use in comparison

task1 = Task(1, "Important")
task2 = Task(2, "Less Important")
print(task1 < task2)  # True - compared by priority only

# Post-initialization processing
@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # Calculated, not in __init__
    
    def __post_init__(self):
        self.area = self.width * self.height

rect = Rectangle(10, 5)
print(rect.area)  # 50

# Inheritance with dataclasses
@dataclass
class Animal:
    name: str
    age: int

@dataclass
class Dog(Animal):
    breed: str

dog = Dog("Buddy", 3, "Golden Retriever")
print(dog)  # Dog(name='Buddy', age=3, breed='Golden Retriever')

# Converting to dict and tuple
from dataclasses import asdict, astuple

person = Person("Bob", 25, "bob@example.com")
person_dict = asdict(person)
print(person_dict)  # {'name': 'Bob', 'age': 25, 'email': 'bob@example.com'}

person_tuple = astuple(person)
print(person_tuple)  # ('Bob', 25, 'bob@example.com')

# Field metadata
@dataclass
class User:
    username: str = field(metadata={"description": "Unique username"})
    password: str = field(repr=False, metadata={"sensitive": True})
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

user = User("john_doe", "secret123")
print(user)  # password not shown in repr

# Replace method (create modified copy)
@dataclass(frozen=True)
class Config:
    host: str
    port: int
    debug: bool

config = Config("localhost", 8080, False)
new_config = config.__class__(config.host, config.port, True)
# Or use replace (if using Python 3.10+)
from dataclasses import replace
# new_config = replace(config, debug=True)

# Complex example: nested dataclasses
@dataclass
class Address:
    street: str
    city: str
    zipcode: str

@dataclass
class Employee:
    name: str
    age: int
    address: Address
    skills: List[str] = field(default_factory=list)
    
    def add_skill(self, skill: str):
        self.skills.append(skill)

address = Address("123 Main St", "Boston", "02101")
employee = Employee("Alice", 30, address)
employee.add_skill("Python")
employee.add_skill("JavaScript")
print(employee)

# When to use dataclasses:
# ✓ Data containers (DTOs, config objects, API models)
# ✓ When you need __repr__, __eq__ automatically
# ✓ Immutable data structures (frozen=True)
# ✓ Simple value objects
# ✗ Complex business logic (use regular classes)
# ✗ When you need custom __init__ logic

---

### Q23: What are Python's magic/dunder methods and when do you use them?

**Answer:**
Magic methods (double underscore methods) let you customize how your classes behave with Python's built-in operations.

```python
# Complete example with most useful magic methods
class Vector:
    def __init__(self, x, y):
        """Constructor"""
        self.x = x
        self.y = y
    
    # String representations
    def __repr__(self):
        """Official string representation - for developers"""
        return f"Vector({self.x}, {self.y})"
    
    def __str__(self):
        """Informal string representation - for users"""
        return f"({self.x}, {self.y})"
    
    # Arithmetic operators
    def __add__(self, other):
        """v1 + v2"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """v1 - v2"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """v * 5"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        """5 * v (reverse multiplication)"""
        return self * scalar
    
    def __truediv__(self, scalar):
        """v / 2"""
        return Vector(self.x / scalar, self.y / scalar)
    
    # Comparison operators
    def __eq__(self, other):
        """v1 == v2"""
        return self.x == other.x and self.y == other.y
    
    def __ne__(self, other):
        """v1 != v2"""
        return not self == other
    
    def __lt__(self, other):
        """v1 < v2 (by magnitude)"""
        return self.magnitude() < other.magnitude()
    
    # Container operations
    def __len__(self):
        """len(v) - returns 2 for 2D vector"""
        return 2
    
    def __getitem__(self, index):
        """v[0] or v[1]"""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Vector index out of range")
    
    def __setitem__(self, index, value):
        """v[0] = 10"""
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        else:
            raise IndexError("Vector index out of range")
    
    # Context manager
    def __enter__(self):
        print(f"Using vector {self}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Done using vector")
        return False
    
    # Callable
    def __call__(self, scalar):
        """v(5) - makes instance callable"""
        return self * scalar
    
    # Boolean context
    def __bool__(self):
        """bool(v) - False if zero vector"""
        return self.x != 0 or self.y != 0
    
    # Hash (for use in sets and as dict keys)
    def __hash__(self):
        """hash(v) - allows use in sets/dict keys"""
        return hash((self.x, self.y))
    
    # Utility
    def magnitude(self):
        return (self.x**2 + self.y**2)**0.5

# Using the Vector class
v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1)  # (3, 4) - uses __str__
print(repr(v1))  # Vector(3, 4) - uses __repr__

v3 = v1 + v2  # Uses __add__
print(v3)  # (4, 6)

v4 = v1 * 2  # Uses __mul__
v5 = 2 * v1  # Uses __rmul__
print(v4, v5)  # Both (6, 8)

print(v1 == Vector(3, 4))  # True - uses __eq__
print(v1 < v2)  # False - uses __lt__

print(len(v1))  # 2 - uses __len__
print(v1[0], v1[1])  # 3 4 - uses __getitem__
v1[0] = 10  # Uses __setitem__

with v1:  # Uses __enter__ and __exit__
    print("Working with vector")

result = v1(3)  # Uses __call__
print(result)  # (30, 12)

if v1:  # Uses __bool__
    print("Non-zero vector")

# Can be used in sets
vectors = {Vector(1, 2), Vector(3, 4), Vector(1, 2)}
print(len(vectors))  # 2 - duplicates removed

# More useful magic methods

# __format__ - custom string formatting
class Money:
    def __init__(self, amount, currency="USD"):
        self.amount = amount
        self.currency = currency
    
    def __format__(self, format_spec):
        if format_spec == 'c':
            symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
            symbol = symbols.get(self.currency, self.currency)
            return f"{symbol}{self.amount:.2f}"
        return str(self.amount)

price = Money(99.99, "USD")
print(f"Price: {price:c}")  # "Price: $99.99"

# __iter__ and __next__ - make class iterable
class Countdown:
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for num in Countdown(5):
    print(num)  # 5, 4, 3, 2, 1

# __contains__ - for 'in' operator
class Inventory:
    def __init__(self):
        self.items = {}
    
    def add(self, item, quantity):
        self.items[item] = quantity
    
    def __contains__(self, item):
        return item in self.items

inventory = Inventory()
inventory.add("apple", 10)
print("apple" in inventory)  # True

# __del__ - destructor (use with caution)
class Resource:
    def __init__(self, name):
        self.name = name
        print(f"Acquiring {name}")
    
    def __del__(self):
        print(f"Releasing {self.name}")
# Note: __del__ is not guaranteed to be called!
# Use context managers instead for cleanup

# __getattr__ and __setattr__ - attribute access
class DynamicAttributes:
    def __getattr__(self, name):
        """Called when attribute not found"""
        return f"'{name}' not found"
    
    def __setattr__(self, name, value):
        """Called on every attribute assignment"""
        print(f"Setting {name} = {value}")
        super().__setattr__(name, value)

obj = DynamicAttributes()
print(obj.nonexistent)  # "'nonexistent' not found"
obj.x = 10  # "Setting x = 10"

---

### Q24: What is method resolution order (MRO) and why does it matter?

**Answer:**
MRO determines the order Python searches for methods in inheritance hierarchies, especially with multiple inheritance.

```python
# Simple inheritance - easy to understand
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")

b = B()
b.method()  # "B" - straightforward

# Multiple inheritance - more complex
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")

class C(A):
    def method(self):
        print("C")

class D(B, C):
    pass

d = D()
d.method()  # "B" - but why?

# Check MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
# Searches in this order: D -> B -> C -> A -> object

# C3 Linearization algorithm
# Rules:
# 1. Child classes come before parents
# 2. Parent order is preserved as specified
# 3. For each class, its parents appear in the same order

# Diamond problem
class Base:
    def __init__(self):
        print("Base.__init__")

class A(Base):
    def __init__(self):
        print("A.__init__")
        super().__init__()

class B(Base):
    def __init__(self):
        print("B.__init__")
        super().__init__()

class C(A, B):
    def __init__(self):
        print("C.__init__")
        super().__init__()

c = C()
# C.__init__
# A.__init__
# B.__init__
# Base.__init__
# Each __init__ called exactly once!

print(C.__mro__)
# (<class 'C'>, <class 'A'>, <class 'B'>, <class 'Base'>, <class 'object'>)

# Why super() is important
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class ColoredRectangle(Rectangle):
    def __init__(self, width, height, color):
        super().__init__(width, height)  # Call parent init
        self.color = color

# Without super() - problems in diamond inheritance
class A:
    def __init__(self):
        self.value_a = "A"

class B(A):
    def __init__(self):
        A.__init__(self)  # Direct call - problematic!
        self.value_b = "B"

class C(A):
    def __init__(self):
        A.__init__(self)  # Direct call - problematic!
        self.value_c = "C"

class D(B, C):
    def __init__(self):
        B.__init__(self)  # A.__init__ called
        C.__init__(self)  # A.__init__ called AGAIN!
# Base class initialized twice!

# With super() - correct
class A:
    def __init__(self):
        print("A init")
        self.value_a = "A"

class B(A):
    def __init__(self):
        super().__init__()  # Follows MRO
        print("B init")
        self.value_b = "B"

class C(A):
    def __init__(self):
        super().__init__()  # Follows MRO
        print("C init")
        self.value_c = "C"

class D(B, C):
    def __init__(self):
        super().__init__()  # Follows MRO
        print("D init")

d = D()
# A init (once!)
# C init
# B init
# D init

# Practical example: Mixins
class JSONMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class TimestampMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from datetime import datetime
        self.created_at = datetime.now()

class User(TimestampMixin, JSONMixin):
    def __init__(self, username, email):
        super().__init__()
        self.username = username
        self.email = email

user = User("john_doe", "john@example.com")
print(user.to_json())  # Includes created_at timestamp

# Checking MRO
print(User.__mro__)

# MRO conflicts - when Python can't determine order
# This will raise TypeError:
# class A: pass
# class B(A): pass
# class C(A): pass
# class D(B, A):  # Error! A already appears before B in MRO
#     pass

# Best practices with MRO:
# 1. Use super() instead of direct parent calls
# 2. Design inheritance hierarchies carefully
# 3. Keep diamond patterns simple
# 4. Use mixins for cross-cutting concerns
# 5. Check MRO with __mro__ when debugging

---

### Q25: What are design patterns in Python and how do you implement them?

**Answer:**
Design patterns are reusable solutions to common problems. Here are the most important ones in Python.

```python
# 1. SINGLETON PATTERN - Single instance of a class
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True

# Better singleton with decorator
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        print("Creating connection")

db1 = DatabaseConnection()  # "Creating connection"
db2 = DatabaseConnection()  # Nothing printed
print(db1 is db2)  # True

# 2. FACTORY PATTERN - Create objects without specifying exact class
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError(f"Unknown animal type: {animal_type}")

# Usage
animal = AnimalFactory.create_animal("dog")
print(animal.speak())  # "Woof!"

# 3. BUILDER PATTERN - Construct complex objects step by step
class Pizza:
    def __init__(self):
        self.dough = None
        self.sauce = None
        self.toppings = []
    
    def __str__(self):
        return f"Pizza with {self.dough} dough, {self.sauce} sauce, toppings: {', '.join(self.toppings)}"

class PizzaBuilder:
    def __init__(self):
        self.pizza = Pizza()
    
    def set_dough(self, dough):
        self.pizza.dough = dough
        return self  # Method chaining
    
    def set_sauce(self, sauce):
        self.pizza.sauce = sauce
        return self
    
    def add_topping(self, topping):
        self.pizza.toppings.append(topping)
        return self
    
    def build(self):
        return self.pizza

# Usage with method chaining
pizza = (PizzaBuilder()
         .set_dough("thin crust")
         .set_sauce("tomato")
         .add_topping("cheese")
         .add_topping("pepperoni")
         .build())
print(pizza)

# 4. OBSERVER PATTERN - Subscribe to and receive notifications
class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self._state)
    
    def set_state(self, state):
        self._state = state
        self.notify()

class Observer:
    def __init__(self, name):
        self.name = name
    
    def update(self, state):
        print(f"{self.name} received update: {state}")

# Usage
subject = Subject()
obs1 = Observer("Observer 1")
obs2 = Observer("Observer 2")

subject.attach(obs1)
subject.attach(obs2)

subject.set_state("New State")
# Observer 1 received update: New State
# Observer 2 received update: New State

# 5. STRATEGY PATTERN - Different algorithms, same interface
class PaymentStrategy:
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} with credit card"

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} with PayPal"

class ShoppingCart:
    def __init__(self, payment_strategy):
        self.payment_strategy = payment_strategy
    
    def checkout(self, amount):
        return self.payment_strategy.pay(amount)

# Usage
cart = ShoppingCart(CreditCardPayment())
print(cart.checkout(100))

cart.payment_strategy = PayPalPayment()
print(cart.checkout(50))

# 6. DECORATOR PATTERN - Add functionality dynamically
class Coffee:
    def cost(self):
        return 5
    
    def description(self):
        return "Coffee"

class MilkDecorator:
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost() + 1
    
    def description(self):
        return self._coffee.description() + ", Milk"

class SugarDecorator:
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost() + 0.5
    
    def description(self):
        return self._coffee.description() + ", Sugar"

# Usage
coffee = Coffee()
coffee = MilkDecorator(coffee)
coffee = SugarDecorator(coffee)
print(f"{coffee.description()}: ${coffee.cost()}")
# "Coffee, Milk, Sugar: $6.5"

# 7. ADAPTER PATTERN - Make incompatible interfaces work together
class EuropeanSocket:
    def provide_electricity(self):
        return "230V"

class USDevice:
    def connect(self, voltage):
        if voltage == "110V":
            return "Device powered"
        return "Voltage mismatch"

class VoltageAdapter:
    def __init__(self, socket):
        self.socket = socket
    
    def provide_power(self):
        eu_voltage = self.socket.provide_electricity()
        # Convert 230V to 110V
        return "110V"

# Usage
socket = EuropeanSocket()
adapter = VoltageAdapter(socket)
device = USDevice()
print(device.connect(adapter.provide_power()))  # "Device powered"

# 8. DEPENDENCY INJECTION - Provide dependencies from outside
# Bad: Hard-coded dependency
class UserServiceBad:
    def __init__(self):
        self.db = MySQLDatabase()  # Tightly coupled!

# Good: Inject dependency
class UserService:
    def __init__(self, database):
        self.db = database  # Can be any database
    
    def get_user(self, user_id):
        return self.db.fetch(user_id)

class MySQLDatabase:
    def fetch(self, id):
        return f"User from MySQL: {id}"

class PostgreSQLDatabase:
    def fetch(self, id):
        return f"User from PostgreSQL: {id}"

# Usage - easy to swap implementations
service = UserService(MySQLDatabase())
print(service.get_user(1))

service = UserService(PostgreSQLDatabase())
print(service.get_user(1))

---

### Q26: How do you implement properties and descriptors in Python?

**Answer:**
Properties and descriptors control attribute access, enabling validation, computed attributes, and more.

```python
# Basic property
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Get temperature in Celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Set temperature in Celsius"""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """Get temperature in Fahrenheit"""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set temperature via Fahrenheit"""
        self.celsius = (value - 32) * 5/9

temp = Temperature(25)
print(temp.celsius)  # 25
print(temp.fahrenheit)  # 77.0

temp.fahrenheit = 32
print(temp.celsius)  # 0.0

# Read-only property (no setter)
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @property
    def area(self):
        return 3.14159 * self.radius ** 2
    
    @property
    def circumference(self):
        return 2 * 3.14159 * self.radius

circle = Circle(5)
print(circle.area)  # 78.53975
# circle.area = 100  # AttributeError: can't set attribute

# Lazy property - computed once
class DataSet:
    def __init__(self, data):
        self._data = data
        self._mean = None
    
    @property
    def mean(self):
        if self._mean is None:
            print("Computing mean...")
            self._mean = sum(self._data) / len(self._data)
        return self._mean

dataset = DataSet([1, 2, 3, 4, 5])
print(dataset.mean)  # "Computing mean..." then 3.0
print(dataset.mean)  # 3.0 (no recomputation)

# Descriptor protocol - for reusable validation
class Validator:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value}")
        instance.__dict__[self.name] = value

class Person:
    age = Validator(min_value=0, max_value=150)
    height = Validator(min_value=0)
    
    def __init__(self, age, height):
        self.age = age
        self.height = height

person = Person(30, 180)
print(person.age)  # 30
# person.age = -5  # ValueError
# person.age = 200  # ValueError

# Type checking descriptor
class TypedProperty:
    def __init__(self, expected_type):
        self.expected_type = expected_type
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be {self.expected_type.__name__}"
            )
        instance.__dict__[self.name] = value

class Product:
    name = TypedProperty(str)
    price = TypedProperty(float)
    quantity = TypedProperty(int)
    
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

product = Product("Laptop", 999.99, 5)
# product.price = "expensive"  # TypeError

# Cached property (Python 3.8+)
from functools import cached_property

class WebPage:
    def __init__(self, url):
        self.url = url
    
    @cached_property
    def content(self):
        print(f"Fetching {self.url}")
        # Expensive operation
        return f"Content from {self.url}"

page = WebPage("https://example.com")
print(page.content)  # "Fetching..." then content
print(page.content)  # Just content (cached)

---

### Q27: What are metaclasses and when should you use them?

**Answer:**
Metaclasses are "classes of classes" - they control class creation. Use them rarely, for frameworks and advanced magic.

```python
# Basic metaclass
class Meta(type):
    def __new__(mcs, name, bases, namespace):
        print(f"Creating class {name}")
        return super().__new__(mcs, name, bases, namespace)

class MyClass(metaclass=Meta):
    pass  # "Creating class MyClass"

# Singleton via metaclass
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("Creating database connection")

db1 = Database()  # "Creating database connection"
db2 = Database()  # Nothing printed
print(db1 is db2)  # True

# Auto-register subclasses
class PluginMeta(type):
    plugins = []
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name != 'Plugin':  # Don't register base class
            mcs.plugins.append(cls)
        return cls

class Plugin(metaclass=PluginMeta):
    pass

class EmailPlugin(Plugin):
    pass

class SMSPlugin(Plugin):
    pass

print(PluginMeta.plugins)  # [EmailPlugin, SMSPlugin]

# When NOT to use metaclasses:
# - Decorators can solve most problems
# - Class decorators are simpler
# - Only use for framework-level magic

# Most people never need metaclasses!
# "Metaclasses are deeper magic than 99% of users should ever worry about" 
# - Tim Peters

---

### Q28: How do you properly handle class and instance attributes?

**Answer:**
Understanding the difference between class and instance attributes is crucial for proper object-oriented design.

```python
class MyClass:
    class_var = []  # Class attribute - shared by all instances
    
    def __init__(self):
        self.instance_var = []  # Instance attribute - unique to each instance

obj1 = MyClass()
obj2 = MyClass()

# Instance attributes are independent
obj1.instance_var.append(1)
print(obj1.instance_var)  # [1]
print(obj2.instance_var)  # []

# Class attributes are shared!
obj1.class_var.append(1)
print(obj1.class_var)  # [1]
print(obj2.class_var)  # [1] - same list!
print(MyClass.class_var)  # [1]

# Proper use of class attributes
class Counter:
    count = 0  # Class attribute for counting instances
    
    def __init__(self, name):
        self.name = name
        Counter.count += 1  # Modify class attribute
    
    @classmethod
    def get_count(cls):
        return cls.count

c1 = Counter("first")
c2 = Counter("second")
print(Counter.get_count())  # 2

# Mutable default argument problem - WRONG
class WrongClass:
    def __init__(self, items=[]):  # BAD!
        self.items = items

obj1 = WrongClass()
obj2 = WrongClass()
obj1.items.append(1)
print(obj2.items)  # [1] - Surprise!

# Correct approach
class CorrectClass:
    def __init__(self, items=None):
        self.items = items if items is not None else []

obj1 = CorrectClass()
obj2 = CorrectClass()
obj1.items.append(1)
print(obj2.items)  # []

# Using class attributes for constants
class Config:
    MAX_CONNECTIONS = 100
    TIMEOUT = 30
    DEFAULT_PORT = 8080

# Name mangling for "private" attributes
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Name mangled to _BankAccount__balance
    
    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
# print(account.__balance)  # AttributeError
print(account._BankAccount__balance)  # 1000 - can still access

---

### Q29: What are slots and when should you use them?

**Answer:**
Slots optimize memory by declaring a fixed set of attributes, preventing the creation of `__dict__`.

```python
# Regular class - uses __dict__
class RegularPerson:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Class with slots - more memory efficient
class SlottedPerson:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Compare memory usage
import sys

regular = RegularPerson("Alice", 30)
slotted = SlottedPerson("Bob", 25)

print(sys.getsizeof(regular.__dict__))  # ~120 bytes
# print(sys.getsizeof(slotted.__dict__))  # AttributeError - no __dict__!

# With slots, you can't add new attributes dynamically
# slotted.email = "bob@example.com"  # AttributeError

# Regular class allows dynamic attributes
regular.email = "alice@example.com"  # Works fine

# Slots with inheritance
class Person:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Employee(Person):
    __slots__ = ['employee_id']  # Additional slots
    
    def __init__(self, name, age, employee_id):
        super().__init__(name, age)
        self.employee_id = employee_id

emp = Employee("Charlie", 35, "E123")

# Memory savings example - creating millions of objects
import time

def measure_memory_and_time(cls, n=1_000_000):
    start_time = time.time()
    objects = [cls(f"Person{i}", i % 100) for i in range(n)]
    end_time = time.time()
    
    # Approximate memory (simplified)
    memory = sys.getsizeof(objects[0]) * n / (1024 * 1024)  # MB
    
    print(f"{cls.__name__}:")
    print(f"  Time: {end_time - start_time:.2f}s")
    print(f"  Approx memory per object: {sys.getsizeof(objects[0])} bytes")

# When to use slots:
# ✓ Creating many instances (millions)
# ✓ Performance-critical code
# ✓ When you know exact attributes needed
# ✗ When you need dynamic attributes
# ✗ When using multiple inheritance (complex)
# ✗ When flexibility is more important than performance

---

### Q30: How do abstract base classes (ABC) work?

**Answer:**
ABCs define interfaces that subclasses must implement, enforcing a contract.

```python
from abc import ABC, abstractmethod

# Define abstract base class
class Shape(ABC):
    @abstractmethod
    def area(self):
        """Calculate area - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate perimeter - must be implemented by subclasses"""
        pass
    
    def describe(self):
        """Concrete method - can be used as-is"""
        return f"Shape with area {self.area()}"

# Cannot instantiate abstract class
# shape = Shape()  # TypeError: Can't instantiate abstract class

# Concrete implementation
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Now we can create instances
rect = Rectangle(5, 3)
print(rect.area())  # 15
print(rect.describe())  # "Shape with area 15"

circle = Circle(5)
print(circle.area())  # 78.53975

# Incomplete implementation fails
class IncompleteShape(Shape):
    def area(self):
        return 0
    # Missing perimeter() implementation

# incomplete = IncompleteShape()  # TypeError

# Abstract properties
class Vehicle(ABC):
    @property
    @abstractmethod
    def wheels(self):
        pass
    
    @abstractmethod
    def drive(self):
        pass

class Car(Vehicle):
    @property
    def wheels(self):
        return 4
    
    def drive(self):
        return "Driving on road"

car = Car()
print(car.wheels)  # 4

# Abstract class methods
class Database(ABC):
    @classmethod
    @abstractmethod
    def connect(cls, connection_string):
        pass
    
    @staticmethod
    @abstractmethod
    def validate_query(query):
        pass

class MySQL(Database):
    @classmethod
    def connect(cls, connection_string):
        return f"Connected to MySQL: {connection_string}"
    
    @staticmethod
    def validate_query(query):
        return "SELECT" in query or "INSERT" in query

# Using ABCs for duck typing validation
from collections.abc import Sized, Iterable

class CustomCollection(Sized, Iterable):
    def __init__(self, items):
        self._items = items
    
    def __len__(self):
        return len(self._items)
    
    def __iter__(self):
        return iter(self._items)

collection = CustomCollection([1, 2, 3])
print(len(collection))  # 3
print(list(collection))  # [1, 2, 3]

# Check if object implements an ABC
print(isinstance(collection, Sized))  # True
print(isinstance(collection, Iterable))  # True

---

### Q31: How do you implement operator overloading effectively?

**Answer:**
Operator overloading lets custom objects work with Python's built-in operators naturally.

```python
class Money:
    def __init__(self, amount, currency="USD"):
        self.amount = amount
        self.currency = currency
    
    def __repr__(self):
        return f"Money({self.amount}, {self.currency})"
    
    def __str__(self):
        return f"${self.amount} {self.currency}"
    
    # Addition
    def __add__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError("Cannot add different currencies")
            return Money(self.amount + other.amount, self.currency)
        elif isinstance(other, (int, float)):
            return Money(self.amount + other, self.currency)
        return NotImplemented
    
    # Right-hand addition (5 + money)
    def __radd__(self, other):
        return self.__add__(other)
    
    # Subtraction
    def __sub__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError("Cannot subtract different currencies")
            return Money(self.amount - other.amount, self.currency)
        elif isinstance(other, (int, float)):
            return Money(self.amount - other, self.currency)
        return NotImplemented
    
    # Multiplication
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Money(self.amount * other, self.currency)
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    # Division
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Money(self.amount / other, self.currency)
        return NotImplemented
    
    # Comparison operators
    def __eq__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount == other.amount and self.currency == other.currency
    
    def __lt__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        if self.currency != other.currency:
            raise ValueError("Cannot compare different currencies")
        return self.amount < other.amount
    
    def __le__(self, other):
        return self == other or self < other
    
    def __gt__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        if self.currency != other.currency:
            raise ValueError("Cannot compare different currencies")
        return self.amount > other.amount
    
    def __ge__(self, other):
        return self == other or self > other

# Usage
m1 = Money(100)
m2 = Money(50)

print(m1 + m2)  # $150 USD
print(m1 - m2)  # $50 USD
print(m1 * 2)   # $200 USD
print(2 * m1)   # $200 USD (uses __rmul__)
print(m1 / 2)   # $50.0 USD

print(m1 > m2)  # True
print(m1 == Money(100))  # True

# Matrix class with operator overloading
class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
    
    def __repr__(self):
        return f"Matrix({self.data})"
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match")
        
        result = [[self.data[i][j] + other.data[i][j] 
                   for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)
    
    def __mul__(self, scalar):
        result = [[self.data[i][j] * scalar 
                   for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)

m1 = Matrix([[1, 2], [3, 4]])
m2 = Matrix([[5, 6], [7, 8]])

m3 = m1 + m2
print(m3)  # Matrix([[6, 8], [10, 12]])

m4 = m1 * 2
print(m4)  # Matrix([[2, 4], [6, 8]])

---

### Q32: What are mixins and how do you use them?

**Answer:**
Mixins are classes that provide specific functionality to be mixed into other classes via multiple inheritance.

```python
# Basic mixin example
class JSONMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_json(cls, json_string):
        import json
        data = json.loads(json_string)
        return cls(**data)

class TimestampMixin:
    def set_timestamp(self):
        from datetime import datetime
        self.created_at = datetime.now().isoformat()

# Using mixins
class User(JSONMixin, TimestampMixin):
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.set_timestamp()

user = User("john_doe", "john@example.com")
json_str = user.to_json()
print(json_str)

# Logging mixin
class LoggingMixin:
    def log(self, message):
        print(f"[{self.__class__.__name__}] {message}")

class Service(LoggingMixin):
    def process(self):
        self.log("Processing started")
        # Do work
        self.log("Processing completed")

service = Service()
service.process()

# Validation mixin
class ValidationMixin:
    def validate(self):
        for attr_name, attr_type in self.__annotations__.items():
            value = getattr(self, attr_name, None)
            if value is None:
                raise ValueError(f"{attr_name} is required")
            if not isinstance(value, attr_type):
                raise TypeError(f"{attr_name} must be {attr_type.__name__}")

class Product(ValidationMixin):
    name: str
    price: float
    quantity: int
    
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity
        self.validate()

product = Product("Laptop", 999.99, 5)
# product = Product("Laptop", "expensive", 5)  # TypeError

# Comparison mixin
class ComparableMixin:
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__
    
    def __ne__(self, other):
        return not self.__eq__(other)

class Person(ComparableMixin):
    def __init__(self, name, age):
        self.name = name
        self.age = age

p1 = Person("Alice", 30)
p2 = Person("Alice", 30)
p3 = Person("Bob", 25)

print(p1 == p2)  # True
print(p1 == p3)  # False

---

### Q33: How do you implement the Iterator and Iterable protocols?

**Answer:**
Implement `__iter__()` and `__next__()` to make objects iterable and work with for loops.

```python
# Basic iterator
class Countdown:
    def __init__(self, start):
        self.start = start
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for num in Countdown(5):
    print(num)  # 5, 4, 3, 2, 1

# Separate iterator and iterable
class NumberRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __iter__(self):
        return NumberRangeIterator(self.start, self.end)

class NumberRangeIterator:
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# Can be iterated multiple times
num_range = NumberRange(1, 5)
print(list(num_range))  # [1, 2, 3, 4]
print(list(num_range))  # [1, 2, 3, 4] - works again

# File reader iterator
class FileReader:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
    
    def __iter__(self):
        self.file = open(self.filename, 'r')
        return self
    
    def __next__(self):
        line = self.file.readline()
        if not line:
            self.file.close()
            raise StopIteration
        return line.strip()

# Usage:
# for line in FileReader('data.txt'):
#     print(line)

# Infinite iterator
class InfiniteCounter:
    def __init__(self, start=0):
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        value = self.current
        self.current += 1
        return value

counter = InfiniteCounter()
# Use with itertools.islice to limit
from itertools import islice
print(list(islice(counter, 5)))  # [0, 1, 2, 3, 4]

---

### Q34: What are protocols and structural subtyping in Python?

**Answer:**
Protocols (Python 3.8+) enable structural typing - objects are compatible if they have the right methods.

```python
from typing import Protocol, runtime_checkable

# Define a protocol
@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> str:
        ...

# Classes don't need to explicitly inherit from Protocol
class Circle:
    def draw(self) -> str:
        return "Drawing a circle"

class Square:
    def draw(self) -> str:
        return "Drawing a square"

# Both work with Drawable protocol
def render(shape: Drawable) -> None:
    print(shape.draw())

circle = Circle()
square = Square()

render(circle)  # Works!
render(square)  # Works!

# Runtime checking
print(isinstance(circle, Drawable))  # True
print(isinstance(square, Drawable))  # True

# Protocol with properties
class Sized(Protocol):
    @property
    def size(self) -> int:
        ...

class Container:
    def __init__(self, items):
        self._items = items
    
    @property
    def size(self) -> int:
        return len(self._items)

container = Container([1, 2, 3])
print(isinstance(container, Sized))  # True

# Multiple method protocol
class Repository(Protocol):
    def save(self, item: object) -> None:
        ...
    
    def find(self, id: int) -> object:
        ...
    
    def delete(self, id: int) -> None:
        ...

# Any class implementing these methods is a Repository
class UserRepository:
    def __init__(self):
        self.users = {}
    
    def save(self, user):
        self.users[user.id] = user
    
    def find(self, id):
        return self.users.get(id)
    
    def delete(self, id):
        del self.users[id]

repo = UserRepository()
print(isinstance(repo, Repository))  # True

---

### Q35: How do you implement polymorphism in Python?

**Answer:**
Python supports polymorphism through duck typing, inheritance, and abstract base classes.

```python
# Duck typing polymorphism
def make_sound(animal):
    # Works with any object that has a speak() method
    return animal.speak()

class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Car:
    def speak(self):
        return "Beep beep!"  # Anything with speak() works

print(make_sound(Dog()))  # "Woof!"
print(make_sound(Cat()))  # "Meow!"
print(make_sound(Car()))  # "Beep beep!"

# Inheritance-based polymorphism
class Shape:
    def area(self):
        raise NotImplementedError

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2

def print_area(shape: Shape):
    print(f"Area: {shape.area()}")

shapes = [Rectangle(5, 3), Circle(4)]
for shape in shapes:
    print_area(shape)

# Operator polymorphism
def add_items(a, b):
    return a + b

print(add_items(5, 3))  # 8 (int addition)
print(add_items("Hello", " World"))  # "Hello World" (string concatenation)
print(add_items([1, 2], [3, 4]))  # [1, 2, 3, 4] (list concatenation)

# Method overriding
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"
    
    def introduce(self):
        return f"{self.name} says: {self.speak()}"

class Dog(Animal):
    def speak(self):  # Override parent method
        return "Woof!"

class Cat(Animal):
    def speak(self):  # Override parent method
        return "Meow!"

animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(animal.introduce())
# Buddy says: Woof!
# Whiskers says: Meow!

---

## Section 3: Functional Programming & Advanced Functions (Q36-50)

### Q36: What is functional programming and how is it applied in Python?

**Answer:**
Functional programming treats computation as the evaluation of mathematical functions, avoiding state and mutable data.

```python
# Pure functions - same input always gives same output, no side effects
def pure_add(a, b):
    return a + b  # No external state modified

# Impure function - has side effects
total = 0
def impure_add(a, b):
    global total
    total += a + b  # Modifies external state
    return total

# First-class functions - functions are objects
def square(x):
    return x ** 2

# Assign to variable
my_func = square
print(my_func(5))  # 25

# Pass as argument
def apply_function(func, value):
    return func(value)

print(apply_function(square, 5))  # 25

# Return from function
def get_operation(op):
    if op == "square":
        return lambda x: x ** 2
    elif op == "double":
        return lambda x: x * 2

operation = get_operation("square")
print(operation(5))  # 25

# Higher-order functions - map, filter, reduce
numbers = [1, 2, 3, 4, 5]

# map - transform each element
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# filter - keep elements that pass test
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]

# reduce - combine elements
from functools import reduce
sum_all = reduce(lambda x, y: x + y, numbers)
print(sum_all)  # 15

# Immutability - prefer creating new data over modifying
original = [1, 2, 3]
# Don't do: original.append(4)
# Do this:
new_list = original + [4]
print(original)  # [1, 2, 3] - unchanged
print(new_list)  # [1, 2, 3, 4]

# Function composition
def add_one(x):
    return x + 1

def double(x):
    return x * 2

def compose(f, g):
    return lambda x: f(g(x))

add_then_double = compose(double, add_one)
print(add_then_double(5))  # 12 (5+1=6, 6*2=12)

# Partial application
from functools import partial

def power(base, exponent):
    return base ** exponent

square_func = partial(power, exponent=2)
cube_func = partial(power, exponent=3)

print(square_func(5))  # 25
print(cube_func(5))   # 125

# Currying - transform function with multiple args to nested functions
def curry_add(a):
    def inner(b):
        return a + b
    return inner

add_5 = curry_add(5)
print(add_5(3))  # 8

# Recursion (functional style for loops)
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # 120

# Tail recursion (Python doesn't optimize this)
def tail_factorial(n, accumulator=1):
    if n <= 1:
        return accumulator
    return tail_factorial(n - 1, n * accumulator)

print(tail_factorial(5))  # 120

---

### Q37: What are lambda functions and when should you use them?

**Answer:**
Lambda functions are anonymous, single-expression functions useful for short operations.

```python
# Basic lambda
square = lambda x: x ** 2
print(square(5))  # 25

# Multiple arguments
add = lambda x, y: x + y
print(add(3, 4))  # 7

# With map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# With filter
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]

# With sorted - custom key
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78}
]
sorted_students = sorted(students, key=lambda s: s["grade"])
print([s["name"] for s in sorted_students])  # ['Charlie', 'Alice', 'Bob']

# With max/min
oldest = max(students, key=lambda s: s["grade"])
print(oldest)  # {'name': 'Bob', 'grade': 92}

# In list comprehensions with conditional
result = [(lambda x: x ** 2)(x) if x % 2 == 0 else x for x in range(10)]
print(result)  # [0, 1, 4, 3, 16, 5, 36, 7, 64, 9]

# Immediately invoked lambda
result = (lambda x, y: x + y)(5, 3)
print(result)  # 8

# Lambda with default arguments
greet = lambda name, greeting="Hello": f"{greeting}, {name}!"
print(greet("Alice"))  # "Hello, Alice!"
print(greet("Bob", "Hi"))  # "Hi, Bob!"

# When NOT to use lambda:
# ✗ Complex logic (use def instead)
# Bad:
complex_lambda = lambda x: x ** 2 if x > 0 else -x if x < 0 else 0

# Good:
def complex_function(x):
    if x > 0:
        return x ** 2
    elif x < 0:
        return -x
    else:
        return 0

# ✗ Assigning to variable (use def for named functions)
# Bad:
my_func = lambda x: x ** 2

# Good:
def my_func(x):
    return x ** 2

# ✓ Good use cases:
# 1. Short callbacks
button.on_click(lambda: print("Button clicked"))

# 2. Key functions
sorted(words, key=lambda w: len(w))

# 3. One-time transformations
list(map(lambda x: x.strip().lower(), lines))

---

### Q38: How do closures work in Python?

**Answer:**
Closures are functions that remember values from their enclosing scope, even after that scope has finished executing.

```python
# Basic closure
def outer(x):
    def inner(y):
        return x + y  # inner() "closes over" x
    return inner

add_5 = outer(5)
print(add_5(3))  # 8
print(add_5(10))  # 15

# The closure remembers x=5
print(add_5.__closure__)  # Cell objects containing closed-over values

# Multiple closures with different values
add_5 = outer(5)
add_10 = outer(10)
print(add_5(3))   # 8
print(add_10(3))  # 13

# Closure for maintaining state
def counter():
    count = 0
    
    def increment():
        nonlocal count  # Modify variable from enclosing scope
        count += 1
        return count
    
    return increment

counter1 = counter()
print(counter1())  # 1
print(counter1())  # 2
print(counter1())  # 3

counter2 = counter()
print(counter2())  # 1 - independent state

# Closure factory
def multiplier(factor):
    return lambda x: x * factor

double = multiplier(2)
triple = multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15

# Practical example: decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
# Hello, Alice!
# Hello, Alice!
# Hello, Alice!

# Closure for private variables
def create_account(initial_balance):
    balance = initial_balance  # Private variable
    
    def deposit(amount):
        nonlocal balance
        balance += amount
        return balance
    
    def withdraw(amount):
        nonlocal balance
        if amount > balance:
            return "Insufficient funds"
        balance -= amount
        return balance
    
    def get_balance():
        return balance
    
    return {
        "deposit": deposit,
        "withdraw": withdraw,
        "get_balance": get_balance
    }

account = create_account(1000)
print(account["deposit"](500))  # 1500
print(account["withdraw"](200))  # 1300
print(account["get_balance"]())  # 1300
# Can't access balance directly - it's encapsulated!

# Closure with loop - common pitfall
# Wrong:
functions = []
for i in range(3):
    functions.append(lambda: i)  # All capture same i!

print([f() for f in functions])  # [2, 2, 2] - all return 2!

# Correct:
functions = []
for i in range(3):
    functions.append(lambda i=i: i)  # Default argument captures current i

print([f() for f in functions])  # [0, 1, 2]

# Or use closure properly:
def make_printer(i):
    return lambda: i

functions = [make_printer(i) for i in range(3)]
print([f() for f in functions])  # [0, 1, 2]

---

### Q39: What are generators and how do they differ from regular functions?

**Answer:**
Generators produce values lazily using `yield`, saving memory and enabling infinite sequences.

```python
# Regular function - returns all at once
def get_numbers_list(n):
    result = []
    for i in range(n):
        result.append(i)
    return result

# Memory intensive for large n
numbers = get_numbers_list(1000000)  # Creates list of 1M items

# Generator - yields one at a time
def get_numbers_generator(n):
    for i in range(n):
        yield i  # Pauses and returns value

# Memory efficient
numbers = get_numbers_generator(1000000)  # Generator object, no list created
print(next(numbers))  # 0
print(next(numbers))  # 1

# Generator with for loop
for num in get_numbers_generator(5):
    print(num)  # 0, 1, 2, 3, 4

# Generator expression (like list comprehension)
squares = (x ** 2 for x in range(10))  # Generator
squares_list = [x ** 2 for x in range(10)]  # List

print(type(squares))  # <class 'generator'>
print(next(squares))  # 0
print(next(squares))  # 1

# Infinite generator
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1

gen = infinite_sequence()
print(next(gen))  # 0
print(next(gen))  # 1
# Can keep going forever!

# Fibonacci generator
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
print([next(fib) for _ in range(10)])
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Generator with state
def counter(start=0):
    count = start
    while True:
        val = yield count
        if val is not None:
            count = val
        else:
            count += 1

c = counter()
print(next(c))  # 0
print(next(c))  # 1
print(c.send(10))  # 10 - reset counter
print(next(c))  # 11

# File processing with generator
def read_large_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# Memory efficient - one line at a time
# for line in read_large_file('large_file.txt'):
#     process(line)

# Generator pipeline
def numbers(n):
    for i in range(n):
        yield i

def squares(nums):
    for num in nums:
        yield num ** 2

def evens(nums):
    for num in nums:
        if num % 2 == 0:
            yield num

# Chain generators
pipeline = evens(squares(numbers(10)))
print(list(pipeline))  # [0, 4, 16, 36, 64]

# Generator with cleanup
def managed_resource(resource_name):
    print(f"Acquiring {resource_name}")
    try:
        yield resource_name
    finally:
        print(f"Releasing {resource_name}")

gen = managed_resource("database")
resource = next(gen)
print(f"Using {resource}")
# gen.close()  # Triggers finally block

# Performance comparison
import sys

# List uses more memory
list_comp = [x ** 2 for x in range(10000)]
print(f"List size: {sys.getsizeof(list_comp)} bytes")

# Generator uses constant memory
gen_expr = (x ** 2 for x in range(10000))
print(f"Generator size: {sys.getsizeof(gen_expr)} bytes")
# Generator size is tiny regardless of sequence length!

---

### Q40: How do you use `map()`, `filter()`, and `reduce()` effectively?

**Answer:**
These functional programming tools transform and combine data without explicit loops.

```python
from functools import reduce

# MAP - Apply function to each element
numbers = [1, 2, 3, 4, 5]

# With lambda
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# With named function
def cube(x):
    return x ** 3

cubed = list(map(cube, numbers))
print(cubed)  # [1, 8, 27, 64, 125]

# Multiple iterables
list1 = [1, 2, 3]
list2 = [10, 20, 30]
result = list(map(lambda x, y: x + y, list1, list2))
print(result)  # [11, 22, 33]

# Map with strings
words = ["hello", "world", "python"]
upper_words = list(map(str.upper, words))
print(upper_words)  # ['HELLO', 'WORLD', 'PYTHON']

# FILTER - Keep elements that pass test
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Get even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# Get numbers > 5
greater_than_5 = list(filter(lambda x: x > 5, numbers))
print(greater_than_5)  # [6, 7, 8, 9, 10]

# Filter strings
words = ["apple", "banana", "avocado", "cherry", "apricot"]
a_words = list(filter(lambda w: w.startswith('a'), words))
print(a_words)  # ['apple', 'avocado', 'apricot']

# Filter with None - removes falsy values
mixed = [0, 1, False, True, "", "hello", None, [], [1, 2]]
truthy = list(filter(None, mixed))
print(truthy)  # [1, True, 'hello', [1, 2]]

# REDUCE - Combine elements into single value
numbers = [1, 2, 3, 4, 5]

# Sum all numbers
total = reduce(lambda x, y: x + y, numbers)
print(total)  # 15

# Product of all numbers
product = reduce(lambda x, y: x * y, numbers)
print(product)  # 120

# Find maximum
maximum = reduce(lambda x, y: x if x > y else y, numbers)
print(maximum)  # 5

# With initial value
total_with_init = reduce(lambda x, y: x + y, numbers, 100)
print(total_with_init)  # 115 (100 + sum of numbers)

# Complex reduce - flatten nested list
nested = [[1, 2], [3, 4], [5, 6]]
flattened = reduce(lambda x, y: x + y, nested)
print(flattened)  # [1, 2, 3, 4, 5, 6]

# Reduce for string concatenation
words = ["Hello", "World", "Python"]
sentence = reduce(lambda x, y: x + " " + y, words)
print(sentence)  # "Hello World Python"

# COMBINING map, filter, reduce
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Square even numbers, then sum them
result = reduce(
    lambda x, y: x + y,
    map(
        lambda x: x ** 2,
        filter(lambda x: x % 2 == 0, numbers)
    )
)
print(result)  # 220 (4 + 16 + 36 + 64 + 100)

# Modern alternative: list comprehension (often more readable)
result = sum(x ** 2 for x in numbers if x % 2 == 0)
print(result)  # 220

# Real-world examples

# 1. Data transformation
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]

# Get names of users over 26
names = list(map(
    lambda u: u["name"],
    filter(lambda u: u["age"] > 26, users)
))
print(names)  # ['Alice', 'Charlie']

# 2. Calculate total price with discount
prices = [10.00, 20.00, 30.00, 40.00]
discount = 0.1

total = reduce(
    lambda x, y: x + y,
    map(lambda p: p * (1 - discount), prices)
)
print(f"Total: ${total:.2f}")  # Total: $90.00

# 3. Parse and validate data
raw_data = ["  123  ", "456", "  789  ", "abc", "012"]

# Strip, filter numeric, convert to int
valid_numbers = list(map(
    int,
    filter(
        lambda x: x.isdigit(),
        map(str.strip, raw_data)
    )
))
print(valid_numbers)  # [123, 456, 789, 12]

# When to use what:
# • map: Transform each element
# • filter: Select elements based on condition
# • reduce: Combine all elements into one value
# • List comprehension: Often more Pythonic and readable

---

### Q41: What is the `functools` module and what are its key functions?

**Answer:**
`functools` provides higher-order functions and operations on callable objects.

```python
from functools import (
    lru_cache, cache, partial, reduce, wraps,
    total_ordering, singledispatch, cached_property
)

# 1. lru_cache - Memoization with Least Recently Used cache
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))  # Fast due to caching
print(fibonacci.cache_info())  # CacheInfo(hits=..., misses=..., maxsize=128, currsize=...)

# Clear cache
fibonacci.cache_clear()

# 2. cache - Unlimited cache (Python 3.9+)
@cache
def expensive_function(x):
    print(f"Computing for {x}")
    return x ** 2

print(expensive_function(5))  # "Computing for 5", then 25
print(expensive_function(5))  # 25 (cached, no print)

# 3. partial - Pre-fill function arguments
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))    # 125

# Practical partial use
from functools import partial
import re

# Create specialized regex functions
find_emails = partial(re.findall, r'\b[\w.-]+@[\w.-]+\.\w+\b')
find_urls = partial(re.findall, r'https?://[\w./]+')

text = "Contact me at user@example.com or visit https://example.com"
print(find_emails(text))
print(find_urls(text))

# 4. wraps - Preserve function metadata in decorators
def my_decorator(func):
    @wraps(func)  # Without this, __name__, __doc__ would be lost
    def wrapper(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper

@my_decorator
def greet(name):
    """Greet someone"""
    return f"Hello, {name}"

print(greet.__name__)  # "greet" (not "wrapper")
print(greet.__doc__)   # "Greet someone"

# 5. reduce - Combine iterable elements
numbers = [1, 2, 3, 4, 5]
sum_all = reduce(lambda x, y: x + y, numbers)
print(sum_all)  # 15

# 6. total_ordering - Generate comparison methods
@total_ordering
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        return self.grade == other.grade
    
    def __lt__(self, other):
        return self.grade < other.grade
    # __le__, __gt__, __ge__ generated automatically!

s1 = Student("Alice", 85)
s2 = Student("Bob", 90)
print(s1 < s2)   # True
print(s1 <= s2)  # True (generated)
print(s1 > s2)   # False (generated)

# 7. singledispatch - Function overloading by type
@singledispatch
def process(arg):
    print(f"Default: {arg}")

@process.register(int)
def _(arg):
    print(f"Processing integer: {arg * 2}")

@process.register(str)
def _(arg):
    print(f"Processing string: {arg.upper()}")

@process.register(list)
def _(arg):
    print(f"Processing list: {len(arg)} items")

process(5)          # "Processing integer: 10"
process("hello")    # "Processing string: HELLO"
process([1, 2, 3])  # "Processing list: 3 items"
process(3.14)       # "Default: 3.14"

# 8. cached_property - Computed once, cached
class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    @cached_property
    def expensive_computation(self):
        print("Computing...")
        return sum(x ** 2 for x in self.data)

processor = DataProcessor([1, 2, 3, 4, 5])
print(processor.expensive_computation)  # "Computing..." then 55
print(processor.expensive_computation)  # 55 (cached)

---

### Q42: How do you use `itertools` for efficient iteration?

**Answer:**
`itertools` provides memory-efficient tools for creating and working with iterators.

```python
import itertools

# 1. count - Infinite counter
counter = itertools.count(start=10, step=2)
print(next(counter))  # 10
print(next(counter))  # 12
print(next(counter))  # 14

# Use with zip to limit
for i, value in zip(itertools.count(), ['a', 'b', 'c']):
    print(i, value)  # 0 a, 1 b, 2 c

# 2. cycle - Infinite cycle through iterable
colors = itertools.cycle(['red', 'green', 'blue'])
print([next(colors) for _ in range(5)])
# ['red', 'green', 'blue', 'red', 'green']

# 3. repeat - Repeat value
repeated = itertools.repeat('A', times=3)
print(list(repeated))  # ['A', 'A', 'A']

# Useful with map
result = list(map(pow, range(5), itertools.repeat(2)))
print(result)  # [0, 1, 4, 9, 16] - squares

# 4. chain - Combine multiple iterables
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = itertools.chain(list1, list2)
print(list(combined))  # [1, 2, 3, 4, 5, 6]

# chain.from_iterable - flatten nested structure
nested = [[1, 2], [3, 4], [5, 6]]
flat = itertools.chain.from_iterable(nested)
print(list(flat))  # [1, 2, 3, 4, 5, 6]

# 5. islice - Slice an iterator
numbers = itertools.count()
first_10 = list(itertools.islice(numbers, 10))
print(first_10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Skip and take
numbers = range(100)
result = list(itertools.islice(numbers, 10, 20, 2))
print(result)  # [10, 12, 14, 16, 18]

# 6. combinations - All combinations
items = ['A', 'B', 'C']
combos = itertools.combinations(items, 2)
print(list(combos))  # [('A', 'B'), ('A', 'C'), ('B', 'C')]

# 7. permutations - All permutations
items = ['A', 'B', 'C']
perms = itertools.permutations(items, 2)
print(list(perms))
# [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# 8. product - Cartesian product
colors = ['red', 'blue']
sizes = ['S', 'M', 'L']
products = itertools.product(colors, sizes)
print(list(products))
# [('red', 'S'), ('red', 'M'), ('red', 'L'), 
#  ('blue', 'S'), ('blue', 'M'), ('blue', 'L')]

# Equivalent to nested loops
for color in colors:
    for size in sizes:
        print(color, size)

# 9. groupby - Group consecutive elements
data = [1, 1, 2, 2, 2, 3, 3, 1, 1]
for key, group in itertools.groupby(data):
    print(key, list(group))
# 1 [1, 1]
# 2 [2, 2, 2]
# 3 [3, 3]
# 1 [1, 1]

# With key function
words = ['apple', 'apricot', 'banana', 'berry', 'cherry']
for key, group in itertools.groupby(words, key=lambda w: w[0]):
    print(f"{key}: {list(group)}")
# a: ['apple', 'apricot']
# b: ['banana', 'berry']
# c: ['cherry']

# 10. takewhile / dropwhile - Take/drop while condition true
numbers = [1, 2, 3, 4, 5, 1, 2, 3]
result = list(itertools.takewhile(lambda x: x < 4, numbers))
print(result)  # [1, 2, 3]

result = list(itertools.dropwhile(lambda x: x < 4, numbers))
print(result)  # [4, 5, 1, 2, 3]

# 11. filterfalse - Opposite of filter
numbers = [1, 2, 3, 4, 5, 6]
odds = list(itertools.filterfalse(lambda x: x % 2 == 0, numbers))
print(odds)  # [1, 3, 5]

# 12. accumulate - Running totals
numbers = [1, 2, 3, 4, 5]
running_sum = list(itertools.accumulate(numbers))
print(running_sum)  # [1, 3, 6, 10, 15]

# With custom operation
running_product = list(itertools.accumulate(numbers, lambda x, y: x * y))
print(running_product)  # [1, 2, 6, 24, 120]

# 13. zip_longest - Zip with fill value
from itertools import zip_longest

list1 = [1, 2, 3]
list2 = ['a', 'b', 'c', 'd', 'e']
result = list(zip_longest(list1, list2, fillvalue=0))
print(result)  # [(1, 'a'), (2, 'b'), (3, 'c'), (0, 'd'), (0, 'e')]

# Real-world examples

# Pagination
def paginate(items, page_size):
    args = [iter(items)] * page_size
    return itertools.zip_longest(*args, fillvalue=None)

data = range(10)
for page in paginate(data, 3):
    print(list(filter(None, page)))  # Remove None values

# Moving window
def sliding_window(iterable, n):
    iterators = itertools.tee(iterable, n)
    for i, it in enumerate(iterators):
        for _ in range(i):
            next(it, None)
    return zip(*iterators)

numbers = [1, 2, 3, 4, 5]
for window in sliding_window(numbers, 3):
    print(window)
# (1, 2, 3)
# (2, 3, 4)
# (3, 4, 5)

---

### Q43: What are comprehensions and how do you use them effectively?

**Answer:**
Comprehensions provide concise syntax for creating lists, sets, dicts, and generators.

```python
# List comprehension
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]

# With if-else (ternary)
result = [x if x % 2 == 0 else -x for x in range(10)]
print(result)  # [0, -1, 2, -3, 4, -5, 6, -7, 8, -9]

# Nested loops
pairs = [(x, y) for x in range(3) for y in range(3)]
print(pairs)
# [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

# Flatten nested list
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [item for sublist in nested for item in sublist]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Set comprehension - removes duplicates
text = "hello world"
unique_chars = {char for char in text if char.isalpha()}
print(unique_chars)  # {'h', 'e', 'l', 'o', 'w', 'r', 'd'}

# Dictionary comprehension
word_lengths = {word: len(word) for word in ['hello', 'world', 'python']}
print(word_lengths)  # {'hello': 5, 'world': 5, 'python': 6}

# Swap keys and values
original = {'a': 1, 'b': 2, 'c': 3}
swapped = {v: k for k, v in original.items()}
print(swapped)  # {1: 'a', 2: 'b', 3: 'c'}

# With condition
prices = {'apple': 0.50, 'banana': 0.30, 'cherry': 0.80}
expensive = {item: price for item, price in prices.items() if price > 0.40}
print(expensive)  # {'apple': 0.5, 'cherry': 0.8}

# Generator expression - memory efficient
squares_gen = (x ** 2 for x in range(1000000))  # No list created!
print(next(squares_gen))  # 0
print(next(squares_gen))  # 1

# Use in functions that accept iterables
total = sum(x ** 2 for x in range(100))  # Generator, not list
print(total)  # 328350

# Real-world examples

# 1. Parse CSV data
csv_lines = ["name,age,city", "Alice,30,NYC", "Bob,25,LA"]
data = [line.split(',') for line in csv_lines[1:]]  # Skip header
print(data)  # [['Alice', '30', 'NYC'], ['Bob', '25', 'LA']]

# 2. Filter and transform
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
doubled_evens = [x * 2 for x in numbers if x % 2 == 0]
print(doubled_evens)  # [4, 8, 12, 16, 20]

# 3. Create lookup dictionary
users = [
    {'id': 1, 'name': 'Alice'},
    {'id': 2, 'name': 'Bob'},
    {'id': 3, 'name': 'Charlie'}
]
user_lookup = {user['id']: user['name'] for user in users}
print(user_lookup)  # {1: 'Alice', 2: 'Bob', 3: 'Charlie'}

# 4. Matrix operations
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# Transpose
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
print(transposed)  # [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

# 5. Filtering dictionary
users = {
    'alice': {'age': 30, 'active': True},
    'bob': {'age': 25, 'active': False},
    'charlie': {'age': 35, 'active': True}
}
active_users = {name: data for name, data in users.items() if data['active']}
print(active_users)

# When NOT to use comprehensions
# Too complex - hurts readability
# Bad:
# result = [process(x) if condition1(x) else alternative(x) 
#           for x in items if condition2(x) and condition3(x)]

# Better as loop:
result = []
for x in items:
    if condition2(x) and condition3(x):
        if condition1(x):
            result.append(process(x))
        else:
            result.append(alternative(x))

# Performance tip: Use generators for large datasets
# Memory heavy:
squares_list = [x ** 2 for x in range(10000000)]  # Creates huge list

# Memory efficient:
squares_gen = (x ** 2 for x in range(10000000))  # Generator object
total = sum(squares_gen)  # Process one at a time

---

### Q44: How do you work with variable-length argument lists?

**Answer:**
Use `*args` for positional arguments and `**kwargs` for keyword arguments to accept variable-length inputs.

```python
# *args - variable positional arguments
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3))  # 6
print(sum_all(1, 2, 3, 4, 5))  # 15

# **kwargs - variable keyword arguments
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30, city="NYC")
# name: Alice
# age: 30
# city: NYC

# Combining regular args, *args, and **kwargs
def complex_function(required, *args, optional=None, **kwargs):
    print(f"Required: {required}")
    print(f"Args: {args}")
    print(f"Optional: {optional}")
    print(f"Kwargs: {kwargs}")

complex_function("must have", 1, 2, 3, optional="yes", extra="data", more="info")
# Required: must have
# Args: (1, 2, 3)
# Optional: yes
# Kwargs: {'extra': 'data', 'more': 'info'}

# Unpacking arguments
def add(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
print(add(*numbers))  # Unpacks list as arguments: add(1, 2, 3)

params = {'a': 1, 'b': 2, 'c': 3}
print(add(**params))  # Unpacks dict as keyword arguments

# Forwarding arguments to another function
def wrapper(*args, **kwargs):
    print("Wrapper called")
    return original_function(*args, **kwargs)

def original_function(x, y, z=10):
    return x + y + z

print(wrapper(1, 2))  # 13
print(wrapper(1, 2, z=20))  # 23

# Keyword-only arguments (after *)
def create_user(name, *, email, age):
    # email and age MUST be passed as keyword arguments
    return {'name': name, 'email': email, 'age': age}

user = create_user("Alice", email="alice@example.com", age=30)
# create_user("Alice", "alice@example.com", 30)  # Error!

# Positional-only arguments (before /) - Python 3.8+
def greet(name, /, greeting="Hello"):
    # name must be positional
    return f"{greeting}, {name}!"

print(greet("Alice"))  # OK
print(greet("Alice", greeting="Hi"))  # OK
# print(greet(name="Alice"))  # Error!

# Combining positional-only, normal, and keyword-only
def func(pos_only, /, pos_or_kwd, *, kwd_only):
    print(f"Positional only: {pos_only}")
    print(f"Positional or keyword: {pos_or_kwd}")
    print(f"Keyword only: {kwd_only}")

func(1, 2, kwd_only=3)  # OK
func(1, pos_or_kwd=2, kwd_only=3)  # OK
# func(pos_only=1, pos_or_kwd=2, kwd_only=3)  # Error!

# Practical examples

# 1. Flexible logging function
def log(level, message, *args, **kwargs):
    formatted_message = message.format(*args)
    metadata = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[{level}] {formatted_message} {metadata}")

log("INFO", "User {} logged in", "Alice", timestamp="2024-01-15", ip="192.168.1.1")
# [INFO] User Alice logged in timestamp=2024-01-15 ip=192.168.1.1

# 2. Function composition
def compose(*functions):
    def inner(x):
        for func in reversed(functions):
            x = func(x)
        return x
    return inner

def add_one(x):
    return x + 1

def double(x):
    return x * 2

def square(x):
    return x ** 2

combined = compose(square, double, add_one)
print(combined(5))  # ((5 + 1) * 2) ** 2 = 144

# 3. Partial application with *args and **kwargs
def partial_right(func, *fixed_args):
    def wrapper(*args):
        return func(*args, *fixed_args)
    return wrapper

def power(base, exponent):
    return base ** exponent

square_func = partial_right(power, 2)
print(square_func(5))  # 25

---

### Q45: What are type hints and how do you use them?

**Answer:**
Type hints (Python 3.5+) provide optional static typing for better code documentation and tooling support.

```python
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

# Basic type hints
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

# Collection types
def process_items(items: List[int]) -> int:
    return sum(items)

def get_user_data() -> Dict[str, Union[str, int]]:
    return {"name": "Alice", "age": 30}

def get_coordinates() -> Tuple[float, float]:
    return (10.5, 20.3)

# Optional - value or None
def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)  # Returns str or None

# Union - multiple types
def process_data(data: Union[int, str, List[int]]) -> str:
    if isinstance(data, int):
        return f"Integer: {data}"
    elif isinstance(data, str):
        return f"String: {data}"
    else:
        return f"List: {data}"

# Any - any type (use sparingly)
def flexible_function(param: Any) -> Any:
    return param

# Callable - function type
def apply_function(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

def multiply(x: int, y: int) -> int:
    return x * y

result = apply_function(multiply, 5, 3)

# Type aliases
UserId = int
UserName = str
UserData = Dict[str, Union[str, int]]

def create_user(user_id: UserId, name: UserName) -> UserData:
    return {"id": user_id, "name": name}

# Generic types
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: List[T] = []
    
    def push(self, item: T) -> None:
        self.items.append(item)
    
    def pop(self) -> T:
        return self.items.pop()

# Type-specific stacks
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)

str_stack: Stack[str] = Stack()
str_stack.push("hello")

# Class type hints
class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name: str = name
        self.age: int = age
    
    def greet(self) -> str:
        return f"Hello, I'm {self.name}"

# Forward references (self-referencing types)
class Node:
    def __init__(self, value: int, next: Optional['Node'] = None) -> None:
        self.value = value
        self.next = next

# Python 3.10+ - Union with | operator
def process(value: int | str) -> str:
    return str(value)

# Python 3.9+ - Built-in generics
def get_items() -> list[str]:  # No need for typing.List
    return ["a", "b", "c"]

def get_mapping() -> dict[str, int]:
    return {"a": 1, "b": 2}

# Literal types - specific values only
from typing import Literal

def set_mode(mode: Literal["read", "write", "append"]) -> None:
    print(f"Mode: {mode}")

set_mode("read")  # OK
# set_mode("delete")  # Type checker warns!

# Type checking with runtime validation
def validate_age(age: int) -> None:
    if not isinstance(age, int):
        raise TypeError("Age must be an integer")
    if age < 0 or age > 150:
        raise ValueError("Invalid age")

# Using mypy for static type checking
# Run: mypy your_script.py

# Example that mypy would catch:
def add_numbers(a: int, b: int) -> int:
    return a + b

# result: str = add_numbers(5, 3)  # mypy error: incompatible types

# TypedDict for structured dictionaries (Python 3.8+)
from typing import TypedDict

class UserDict(TypedDict):
    name: str
    age: int
    email: str

def process_user(user: UserDict) -> str:
    return f"{user['name']} ({user['age']})"

# Protocol for structural subtyping
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

def render(obj: Drawable) -> None:
    print(obj.draw())

class Circle:
    def draw(self) -> str:
        return "Drawing circle"

render(Circle())  # Type checker approves!

# Benefits of type hints:
# 1. Better IDE autocomplete and error detection
# 2. Self-documenting code
# 3. Catch bugs before runtime
# 4. Easier refactoring
# 5. Better collaboration in teams

---

### Q46: How do you use Python's `operator` module?

**Answer:**
The `operator` module provides function equivalents for Python's operators, useful in functional programming.

```python
import operator

# Arithmetic operators
print(operator.add(5, 3))  # 8 (same as 5 + 3)
print(operator.sub(5, 3))  # 2 (same as 5 - 3)
print(operator.mul(5, 3))  # 15 (same as 5 * 3)
print(operator.truediv(10, 2))  # 5.0 (same as 10 / 2)
print(operator.floordiv(10, 3))  # 3 (same as 10 // 3)
print(operator.mod(10, 3))  # 1 (same as 10 % 3)
print(operator.pow(2, 3))  # 8 (same as 2 ** 3)

# Comparison operators
print(operator.eq(5, 5))  # True (same as 5 == 5)
print(operator.ne(5, 3))  # True (same as 5 != 3)
print(operator.lt(3, 5))  # True (same as 3 < 5)
print(operator.le(5, 5))  # True (same as 5 <= 5)
print(operator.gt(5, 3))  # True (same as 5 > 3)
print(operator.ge(5, 5))  # True (same as 5 >= 5)

# Logical operators
print(operator.and_(True, False))  # False (same as True and False)
print(operator.or_(True, False))  # True (same as True or False)
print(operator.not_(True))  # False (same as not True)

# Using with higher-order functions
numbers = [1, 2, 3, 4, 5]

# Instead of lambda
from functools import reduce
total = reduce(operator.add, numbers)
print(total)  # 15

# Sort by specific field
students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Charlie', 'grade': 78}
]

# Using operator.itemgetter (better than lambda)
sorted_students = sorted(students, key=operator.itemgetter('grade'))
print([s['name'] for s in sorted_students])  # ['Charlie', 'Alice', 'Bob']

# Multiple keys
data = [
    ('Alice', 30, 'NYC'),
    ('Bob', 25, 'LA'),
    ('Alice', 25, 'Boston')
]
sorted_data = sorted(data, key=operator.itemgetter(0, 1))
print(sorted_data)
# [('Alice', 25, 'Boston'), ('Alice', 30, 'NYC'), ('Bob', 25, 'LA')]

# operator.attrgetter - get attributes
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

people = [Person('Alice', 30), Person('Bob', 25), Person('Charlie', 35)]
sorted_people = sorted(people, key=operator.attrgetter('age'))
print([p.name for p in sorted_people])  # ['Bob', 'Alice', 'Charlie']

# Multiple attributes
sorted_people = sorted(people, key=operator.attrgetter('name', 'age'))

# operator.methodcaller - call method
strings = ['hello', 'world', 'python']
upper_strings = list(map(operator.methodcaller('upper'), strings))
print(upper_strings)  # ['HELLO', 'WORLD', 'PYTHON']

# With arguments
strings = ['  hello  ', '  world  ']
trimmed = list(map(operator.methodcaller('strip'), strings))
print(trimmed)  # ['hello', 'world']

# Replace value
texts = ['hello world', 'foo bar']
replaced = list(map(operator.methodcaller('replace', 'o', '0'), texts))
print(replaced)  # ['hell0 w0rld', 'f00 bar']

# In-place operators
numbers = [1, 2, 3]
operator.iadd(numbers, [4, 5])  # Same as numbers += [4, 5]
print(numbers)  # [1, 2, 3, 4, 5]

# Practical examples

# 1. Max/min by specific field
products = [
    {'name': 'Apple', 'price': 1.50},
    {'name': 'Banana', 'price': 0.75},
    {'name': 'Cherry', 'price': 2.00}
]

cheapest = min(products, key=operator.itemgetter('price'))
print(cheapest)  # {'name': 'Banana', 'price': 0.75}

# 2. Group by field
from itertools import groupby

data = [
    {'category': 'fruit', 'name': 'apple'},
    {'category': 'fruit', 'name': 'banana'},
    {'category': 'vegetable', 'name': 'carrot'},
    {'category': 'vegetable', 'name': 'lettuce'}
]

data_sorted = sorted(data, key=operator.itemgetter('category'))
for category, items in groupby(data_sorted, key=operator.itemgetter('category')):
    print(f"{category}: {[item['name'] for item in items]}")

# 3. Chaining comparisons
def compare_tuples(a, b):
    return operator.eq(a, b)

print(compare_tuples((1, 2), (1, 2)))  # True

---

### Q47: What are first-class functions and how do you leverage them?

**Answer:**
First-class functions can be assigned to variables, passed as arguments, and returned from functions.

```python
# Functions as variables
def greet(name):
    return f"Hello, {name}!"

# Assign to variable
say_hello = greet
print(say_hello("Alice"))  # "Hello, Alice!"

# Store in data structures
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

operations = {
    'add': add,
    'subtract': subtract,
    'multiply': multiply
}

result = operations['add'](5, 3)
print(result)  # 8

# Functions as arguments (higher-order functions)
def apply_operation(func, a, b):
    return func(a, b)

print(apply_operation(add, 10, 5))  # 15
print(apply_operation(multiply, 10, 5))  # 50

# Functions as return values
def create_multiplier(factor):
    def multiplier(x):
        return x * factor
    return multiplier

double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15

# Practical example: Strategy pattern
class PaymentProcessor:
    def __init__(self, payment_method):
        self.payment_method = payment_method
    
    def process(self, amount):
        return self.payment_method(amount)

def credit_card_payment(amount):
    return f"Paid ${amount} with credit card"

def paypal_payment(amount):
    return f"Paid ${amount} with PayPal"

processor = PaymentProcessor(credit_card_payment)
print(processor.process(100))

processor.payment_method = paypal_payment
print(processor.process(50))

# Function factory with configuration
def create_validator(min_val, max_val):
    def validator(value):
        return min_val <= value <= max_val
    return validator

age_validator = create_validator(0, 150)
temperature_validator = create_validator(-273, 1000)

print(age_validator(30))  # True
print(age_validator(200))  # False
print(temperature_validator(25))  # True

# Callback functions
def process_data(data, callback):
    result = [x * 2 for x in data]
    callback(result)

def print_result(result):
    print(f"Result: {result}")

process_data([1, 2, 3], print_result)

# Function registry pattern
_handlers = {}

def register_handler(event_type):
    def decorator(func):
        _handlers[event_type] = func
        return func
    return decorator

@register_handler('click')
def handle_click():
    print("Click handled")

@register_handler('submit')
def handle_submit():
    print("Submit handled")

# Dispatch based on event
def dispatch_event(event_type):
    handler = _handlers.get(event_type)
    if handler:
        handler()

dispatch_event('click')  # "Click handled"

---

### Q48: How do you use recursion effectively in Python?

**Answer:**
Recursion solves problems by having functions call themselves with simpler inputs.

```python
# Basic recursion - factorial
def factorial(n):
    if n <= 1:  # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

print(factorial(5))  # 120

# Fibonacci sequence
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))  # 55

# Problem: fibonacci is inefficient (exponential time)
# Solution: memoization
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

print(fibonacci_cached(100))  # Fast!

# Tree traversal
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def inorder_traversal(node):
    if node is None:
        return []
    
    result = []
    result.extend(inorder_traversal(node.left))
    result.append(node.value)
    result.extend(inorder_traversal(node.right))
    return result

# Build tree
root = TreeNode(1,
    TreeNode(2, TreeNode(4), TreeNode(5)),
    TreeNode(3)
)

print(inorder_traversal(root))  # [4, 2, 5, 1, 3]

# Deep copy with recursion
def deep_copy(obj):
    if isinstance(obj, dict):
        return {k: deep_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_copy(item) for item in obj]
    else:
        return obj

original = {'a': [1, 2, {'b': 3}]}
copied = deep_copy(original)
copied['a'][2]['b'] = 999
print(original)  # Unchanged

# Flatten nested structure
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

nested = [1, [2, [3, 4], 5], 6, [7, 8]]
print(flatten(nested))  # [1, 2, 3, 4, 5, 6, 7, 8]

# Tail recursion (Python doesn't optimize this)
def tail_factorial(n, accumulator=1):
    if n <= 1:
        return accumulator
    return tail_factorial(n - 1, n * accumulator)

print(tail_factorial(5))  # 120

# Recursion depth limit
import sys
print(sys.getrecursionlimit())  # Usually 1000

# Increase if needed (use cautiously)
# sys.setrecursionlimit(10000)

# Iterative alternative (preferred for deep recursion)
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# When to use recursion:
# ✓ Tree/graph traversal
# ✓ Divide and conquer algorithms
# ✓ Backtracking problems
# ✓ When problem naturally recursive
# ✗ Simple loops (use iteration)
# ✗ Deep recursion (stack overflow risk)
# ✗ Performance-critical code (recursion overhead)

---

### Q49: What is partial application and function currying?

**Answer:**
Partial application fixes some arguments of a function, currying transforms multi-argument functions into chains of single-argument functions.

```python
from functools import partial

# Partial application
def power(base, exponent):
    return base ** exponent

# Create specialized functions
square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))    # 125

# Practical partial application
import re

# Create specialized regex functions
find_emails = partial(re.findall, r'\b[\w.-]+@[\w.-]+\.\w+\b')
find_numbers = partial(re.findall, r'\d+')

text = "Contact me at user@example.com or call 555-1234"
print(find_emails(text))  # ['user@example.com']
print(find_numbers(text))  # ['555', '1234']

# Manual currying
def curry_add(a):
    def inner(b):
        return a + b
    return inner

add_5 = curry_add(5)
print(add_5(3))  # 8
print(add_5(10))  # 15

# Curry with multiple arguments
def curry_multiply(a):
    def inner1(b):
        def inner2(c):
            return a * b * c
        return inner2
    return inner1

result = curry_multiply(2)(3)(4)
print(result)  # 24

# Generic currying function
def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))
    return curried

@curry
def add_three(a, b, c):
    return a + b + c

print(add_three(1)(2)(3))  # 6
print(add_three(1, 2)(3))  # 6
print(add_three(1, 2, 3))  # 6

# Real-world example: logging with context
def create_logger(level):
    def logger(module):
        def log(message):
            print(f"[{level}] [{module}] {message}")
        return log
    return logger

info_logger = create_logger("INFO")
auth_logger = info_logger("AUTH")
api_logger = info_logger("API")

auth_logger("User logged in")  # [INFO] [AUTH] User logged in
api_logger("Request received")  # [INFO] [API] Request received

# Partial with operator module
from operator import add, mul

add_10 = partial(add, 10)
print(add_10(5))  # 15

double = partial(mul, 2)
print(double(5))  # 10

---

### Q50: How do you use `zip()` and its advanced patterns?

**Answer:**
`zip()` combines multiple iterables into tuples, enabling parallel iteration and data transformation.

```python
# Basic zip
names = ['Alice', 'Bob', 'Charlie']
ages = [30, 25, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Create dictionary from two lists
keys = ['name', 'age', 'city']
values = ['Alice', 30, 'NYC']
person = dict(zip(keys, values))
print(person)  # {'name': 'Alice', 'age': 30, 'city': 'NYC'}

# Zip stops at shortest sequence
list1 = [1, 2, 3, 4, 5]
list2 = ['a', 'b', 'c']
print(list(zip(list1, list2)))  # [(1, 'a'), (2, 'b'), (3, 'c')]

# zip_longest - continues to longest
from itertools import zip_longest

result = list(zip_longest(list1, list2, fillvalue='X'))
print(result)  # [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'X'), (5, 'X')]

# Unzipping
pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
numbers, letters = zip(*pairs)  # Unpack with *
print(numbers)  # (1, 2, 3)
print(letters)  # ('a', 'b', 'c')

# Transpose matrix
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
transposed = list(zip(*matrix))
print(transposed)
# [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

# Parallel iteration with enumerate
names = ['Alice', 'Bob', 'Charlie']
scores = [85, 92, 78]

for i, (name, score) in enumerate(zip(names, scores), start=1):
    print(f"{i}. {name}: {score}")

# Comparing adjacent elements
numbers = [1, 3, 2, 5, 4, 6]
for current, next_val in zip(numbers, numbers[1:]):
    print(f"{current} -> {next_val}")

# Sliding window
def sliding_window(lst, size):
    iterators = [iter(lst[i:]) for i in range(size)]
    return zip(*iterators)

numbers = [1, 2, 3, 4, 5]
for window in sliding_window(numbers, 3):
    print(window)
# (1, 2, 3)
# (2, 3, 4)
# (3, 4, 5)

# Merge sorted lists
list1 = [1, 3, 5]
list2 = [2, 4, 6]
merged = sorted(list1 + list2)  # Simple way

# Or with zip for paired operations
for a, b in zip(sorted(list1), sorted(list2)):
    print(f"Pair: {a}, {b}")

---

## Section 4: Iterators, Generators & Memory Optimization (Q51-65)

### Q51: How do you create memory-efficient data pipelines?

**Answer:**
Use generators and iterators to process data one item at a time without loading everything into memory.

```python
# Memory inefficient - loads all into memory
def process_file_bad(filename):
    with open(filename) as f:
        lines = f.readlines()  # All lines in memory!
    
    processed = []
    for line in lines:
        processed.append(line.strip().upper())
    return processed

# Memory efficient - generator pipeline
def read_lines(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()

def filter_empty(lines):
    for line in lines:
        if line:
            yield line

def to_upper(lines):
    for line in lines:
        yield line.upper()

# Chain generators
def process_file_good(filename):
    pipeline = to_upper(filter_empty(read_lines(filename)))
    return pipeline

# Usage - processes one line at a time
# for line in process_file_good('large_file.txt'):
#     process(line)

# Generator expressions in pipeline
def process_data_pipeline(numbers):
    # Each step is a generator
    step1 = (x * 2 for x in numbers)
    step2 = (x for x in step1 if x > 10)
    step3 = (x ** 2 for x in step2)
    return step3

numbers = range(100)
result = process_data_pipeline(numbers)
print(list(result)[:5])  # Only compute what's needed

# CSV processing without pandas
def read_csv(filename):
    with open(filename) as f:
        header = next(f).strip().split(',')
        for line in f:
            values = line.strip().split(',')
            yield dict(zip(header, values))

def filter_rows(rows, condition):
    for row in rows:
        if condition(row):
            yield row

def transform_rows(rows, transform_func):
    for row in rows:
        yield transform_func(row)

# Example usage
# rows = read_csv('data.csv')
# filtered = filter_rows(rows, lambda r: int(r['age']) > 25)
# transformed = transform_rows(filtered, lambda r: {**r, 'age_group': 'adult'})

# Memory comparison
import sys

# List - all in memory
large_list = [x ** 2 for x in range(1000000)]
print(f"List memory: {sys.getsizeof(large_list)} bytes")

# Generator - constant memory
large_gen = (x ** 2 for x in range(1000000))
print(f"Generator memory: {sys.getsizeof(large_gen)} bytes")

# Chunked processing
def process_in_chunks(iterable, chunk_size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

# Process large dataset in manageable chunks
numbers = range(1000)
for chunk in process_in_chunks(numbers, 100):
    # Process 100 items at a time
    result = sum(chunk)
    print(f"Chunk sum: {result}")

---

### Q52: What is the difference between `__iter__` and `__next__`?

**Answer:**
`__iter__()` returns the iterator object itself, `__next__()` returns the next value or raises StopIteration.

```python
# Iterator protocol
class CountDown:
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        # Return iterator object (self)
        return self
    
    def __next__(self):
        # Return next value or raise StopIteration
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

countdown = CountDown(5)
for num in countdown:
    print(num)  # 5, 4, 3, 2, 1

# Separate iterator from iterable
class NumberRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __iter__(self):
        return NumberRangeIterator(self.start, self.end)

class NumberRangeIterator:
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# Can iterate multiple times
num_range = NumberRange(1, 5)
print(list(num_range))  # [1, 2, 3, 4]
print(list(num_range))  # [1, 2, 3, 4] - works again!

# Manual iteration
iterator = iter([1, 2, 3])
print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
# print(next(iterator))  # StopIteration

# With default value
print(next(iterator, "Done"))  # "Done" instead of StopIteration

---

### Q53: How do you use `yield from` for generator delegation?

**Answer:**
`yield from` delegates to a sub-generator, simplifying nested generator patterns.

```python
# Without yield from - manual delegation
def flatten_manual(nested):
    for sublist in nested:
        for item in sublist:
            yield item

# With yield from - cleaner
def flatten(nested):
    for sublist in nested:
        yield from sublist

nested = [[1, 2], [3, 4], [5, 6]]
print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6]

# Recursive flattening
def deep_flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from deep_flatten(item)  # Recursive delegation
        else:
            yield item

deeply_nested = [1, [2, [3, 4], 5], 6, [7, 8]]
print(list(deep_flatten(deeply_nested)))  # [1, 2, 3, 4, 5, 6, 7, 8]

# Tree traversal
class TreeNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []
    
    def traverse(self):
        yield self.value
        for child in self.children:
            yield from child.traverse()

root = TreeNode(1, [
    TreeNode(2, [TreeNode(4), TreeNode(5)]),
    TreeNode(3)
])

print(list(root.traverse()))  # [1, 2, 4, 5, 3]

# Generator chaining
def read_files(*filenames):
    for filename in filenames:
        yield from read_file(filename)

def read_file(filename):
    with open(filename) as f:
        yield from f

# Processes all files as one stream
# for line in read_files('file1.txt', 'file2.txt'):
#     process(line)

---

### Q54: What are itertools recipes and commonly used patterns?

**Answer:**
The itertools documentation provides powerful combination patterns for common iteration tasks.

```python
import itertools

# Pairwise iteration (Python 3.10+ has built-in)
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

numbers = [1, 2, 3, 4, 5]
for a, b in pairwise(numbers):
    print(f"{a} -> {b}")

# Chunking
def chunked(iterable, n):
    it = iter(iterable)
    while chunk := list(itertools.islice(it, n)):
        yield chunk

for chunk in chunked(range(10), 3):
    print(chunk)
# [0, 1, 2]
# [3, 4, 5]
# [6, 7, 8]
# [9]

# Take n items
def take(n, iterable):
    return list(itertools.islice(iterable, n))

print(take(5, itertools.count()))  # [0, 1, 2, 3, 4]

# First true value
def first_true(iterable, default=None, predicate=None):
    return next(filter(predicate, iterable), default)

numbers = [0, 0, 3, 5, 0]
print(first_true(numbers, predicate=lambda x: x > 2))  # 3

# Unique elements preserving order
def unique_everseen(iterable, key=None):
    seen = set()
    for element in iterable:
        k = element if key is None else key(element)
        if k not in seen:
            seen.add(k)
            yield element

data = [1, 2, 2, 3, 1, 4, 5, 3]
print(list(unique_everseen(data)))  # [1, 2, 3, 4, 5]

# Flatten one level
def flatten_one_level(list_of_lists):
    return itertools.chain.from_iterable(list_of_lists)

nested = [[1, 2], [3, 4], [5]]
print(list(flatten_one_level(nested)))  # [1, 2, 3, 4, 5]

# Partition - split into true/false groups
def partition(predicate, iterable):
    t1, t2 = itertools.tee(iterable)
    return (
        filter(predicate, t1),
        itertools.filterfalse(predicate, t2)
    )

numbers = range(10)
evens, odds = partition(lambda x: x % 2 == 0, numbers)
print(list(evens))  # [0, 2, 4, 6, 8]
print(list(odds))   # [1, 3, 5, 7, 9]

# Sliding window
def sliding_window(iterable, n):
    it = iter(iterable)
    window = list(itertools.islice(it, n))
    if len(window) == n:
        yield tuple(window)
    for item in it:
        window = window[1:] + [item]
        yield tuple(window)

for window in sliding_window(range(6), 3):
    print(window)
# (0, 1, 2)
# (1, 2, 3)
# (2, 3, 4)
# (3, 4, 5)

---

### Q55: How do you implement lazy evaluation in Python?

**Answer:**
Lazy evaluation defers computation until the value is actually needed, saving memory and time.

```python
# Eager evaluation - computes immediately
def eager_range(n):
    result = []
    for i in range(n):
        result.append(i ** 2)  # All computed upfront
    return result

# Lazy evaluation - computes on demand
def lazy_range(n):
    for i in range(n):
        yield i ** 2  # Computed only when requested

# Memory comparison
import sys
eager = eager_range(1000000)  # Large memory footprint
lazy = lazy_range(1000000)    # Tiny memory footprint

print(f"Eager: {sys.getsizeof(eager)} bytes")
print(f"Lazy: {sys.getsizeof(lazy)} bytes")

# Lazy property evaluation
class ExpensiveComputation:
    def __init__(self, data):
        self._data = data
        self._result = None
    
    @property
    def result(self):
        if self._result is None:
            print("Computing result...")
            self._result = sum(x ** 2 for x in self._data)
        return self._result

obj = ExpensiveComputation([1, 2, 3, 4, 5])
# No computation yet
print("Created object")
print(obj.result)  # "Computing result..." then 55
print(obj.result)  # 55 (cached)

# Lazy sequences with itertools
import itertools

# Infinite lazy sequences
naturals = itertools.count(1)  # 1, 2, 3, ...
squares = (x ** 2 for x in naturals)  # 1, 4, 9, ...

# Only computed when requested
print(next(squares))  # 1
print(next(squares))  # 4

# Short-circuit evaluation
def expensive_check():
    print("Expensive check called")
    return True

# 'and' short-circuits if first is False
if False and expensive_check():
    pass  # expensive_check() never called

# 'or' short-circuits if first is True
if True or expensive_check():
    pass  # expensive_check() never called

# Lazy data loading
class LazyDataLoader:
    def __init__(self, filename):
        self.filename = filename
        self._data = None
    
    @property
    def data(self):
        if self._data is None:
            print(f"Loading {self.filename}...")
            # Simulate expensive load
            self._data = [1, 2, 3, 4, 5]
        return self._data

loader = LazyDataLoader("data.csv")
# File not loaded yet
print("Loader created")
print(loader.data)  # "Loading data.csv..." then data
print(loader.data)  # Just data (cached)

# Lazy decorator
def lazy_property(func):
    attr_name = '_lazy_' + func.__name__
    
    @property
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return wrapper

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
    
    @lazy_property
    def mean(self):
        print("Computing mean...")
        return sum(self.data) / len(self.data)
    
    @lazy_property
    def variance(self):
        print("Computing variance...")
        mean = self.mean
        return sum((x - mean) ** 2 for x in self.data) / len(self.data)

analyzer = DataAnalyzer([1, 2, 3, 4, 5])
print(analyzer.mean)      # "Computing mean..." then 3.0
print(analyzer.mean)      # 3.0 (cached)
print(analyzer.variance)  # "Computing variance..." then 2.0

---

### Q56: How do you handle large files without loading them into memory?

**Answer:**
Use iterators and generators to process files line-by-line or in chunks.

```python
# Bad - loads entire file into memory
def process_file_bad(filename):
    with open(filename) as f:
        content = f.read()  # Entire file in memory!
    return content.upper()

# Good - line-by-line processing
def process_file_good(filename):
    with open(filename) as f:
        for line in f:  # Iterator - one line at a time
            yield line.strip().upper()

# Binary file - chunk processing
def read_in_chunks(filename, chunk_size=1024):
    with open(filename, 'rb') as f:
        while chunk := f.read(chunk_size):
            yield chunk

# for chunk in read_in_chunks('large_file.bin'):
#     process(chunk)

# CSV processing without pandas
def process_csv(filename):
    with open(filename) as f:
        header = next(f).strip().split(',')
        for line in f:
            values = line.strip().split(',')
            yield dict(zip(header, values))

# for row in process_csv('data.csv'):
#     if int(row['age']) > 25:
#         print(row['name'])

# Log file analysis
def analyze_log(filename, pattern):
    import re
    with open(filename) as f:
        for line in f:
            if re.search(pattern, line):
                yield line.strip()

# errors = analyze_log('app.log', r'ERROR')
# for error in errors:
#     handle_error(error)

# Counting without loading
def count_lines(filename):
    count = 0
    with open(filename) as f:
        for _ in f:
            count += 1
    return count

# More efficient for very large files
def count_lines_buffered(filename):
    with open(filename, 'rb') as f:
        count = sum(chunk.count(b'\n') for chunk in iter(lambda: f.read(1024 * 1024), b''))
    return count

# Search and replace streaming
def search_and_replace(input_file, output_file, search, replace):
    with open(input_file) as infile, open(output_file, 'w') as outfile:
        for line in infile:
            outfile.write(line.replace(search, replace))

# Merge sorted files
def merge_sorted_files(*filenames):
    import heapq
    
    def file_iterator(filename):
        with open(filename) as f:
            for line in f:
                yield int(line.strip())
    
    iterators = [file_iterator(fn) for fn in filenames]
    for value in heapq.merge(*iterators):
        yield value

# Parallel processing of large file
from multiprocessing import Pool

def process_chunk(chunk):
    # Process chunk of lines
    return [line.upper() for line in chunk]

def parallel_process_file(filename, num_workers=4):
    with open(filename) as f:
        lines = f.readlines()
    
    chunk_size = len(lines) // num_workers
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    with Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunks)
    
    return [item for chunk in results for item in chunk]

---

### Q57: What are generator-based coroutines and how do they work?

**Answer:**
Generator-based coroutines use `yield` to both produce and consume values, enabling bidirectional communication.

```python
# Basic coroutine
def simple_coroutine():
    print("Coroutine started")
    x = yield  # Wait for value
    print(f"Received: {x}")

coro = simple_coroutine()
next(coro)  # Prime the coroutine - "Coroutine started"
coro.send(10)  # "Received: 10"

# Coroutine that receives and returns
def averager():
    total = 0
    count = 0
    average = None
    while True:
        value = yield average  # Return average, receive new value
        total += value
        count += 1
        average = total / count

avg = averager()
next(avg)  # Prime it
print(avg.send(10))  # 10.0
print(avg.send(20))  # 15.0
print(avg.send(30))  # 20.0

# Pipeline with coroutines
def producer(consumer):
    for i in range(5):
        consumer.send(i)
    consumer.close()

def filter_even(next_stage):
    try:
        while True:
            value = yield
            if value % 2 == 0:
                next_stage.send(value)
    except GeneratorExit:
        next_stage.close()

def printer():
    try:
        while True:
            value = yield
            print(f"Received: {value}")
    except GeneratorExit:
        print("Printer closed")

# Setup pipeline
p = printer()
next(p)
f = filter_even(p)
next(f)
producer(f)
# Received: 0
# Received: 2
# Received: 4
# Printer closed

# Coroutine decorator
def coroutine(func):
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)  # Prime automatically
        return gen
    return wrapper

@coroutine
def running_sum():
    total = 0
    while True:
        value = yield total
        total += value

rs = running_sum()  # Already primed!
print(rs.send(10))  # 10
print(rs.send(20))  # 30
print(rs.send(5))   # 35

---

### Q58: How do you implement the Iterator design pattern efficiently?

**Answer:**
The Iterator pattern provides sequential access to elements without exposing internal structure.

```python
# Custom collection with iterator
class BookCollection:
    def __init__(self):
        self.books = []
    
    def add_book(self, book):
        self.books.append(book)
    
    def __iter__(self):
        return BookIterator(self.books)

class BookIterator:
    def __init__(self, books):
        self.books = books
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.books):
            raise StopIteration
        book = self.books[self.index]
        self.index += 1
        return book

collection = BookCollection()
collection.add_book("Book 1")
collection.add_book("Book 2")
collection.add_book("Book 3")

for book in collection:
    print(book)

# Reverse iterator
class ReverseIterator:
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]

for item in ReverseIterator([1, 2, 3, 4]):
    print(item)  # 4, 3, 2, 1

# Filter iterator
class FilterIterator:
    def __init__(self, iterable, predicate):
        self.iterator = iter(iterable)
        self.predicate = predicate
    
    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            item = next(self.iterator)
            if self.predicate(item):
                return item

numbers = range(10)
evens = FilterIterator(numbers, lambda x: x % 2 == 0)
print(list(evens))  # [0, 2, 4, 6, 8]

# Tree iterator (in-order traversal)
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
    
    def __iter__(self):
        return TreeIterator(self)

class TreeIterator:
    def __init__(self, root):
        self.stack = []
        self._push_left(root)
    
    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.stack:
            raise StopIteration
        
        node = self.stack.pop()
        self._push_left(node.right)
        return node.value

root = TreeNode(2,
    TreeNode(1),
    TreeNode(3)
)

for value in root:
    print(value)  # 1, 2, 3

---

### Q59: How do you use `itertools.tee()` for multiple independent iterators?

**Answer:**
`tee()` splits one iterator into multiple independent iterators for parallel processing.

```python
import itertools

# Basic tee usage
numbers = range(5)
it1, it2 = itertools.tee(numbers, 2)

print(list(it1))  # [0, 1, 2, 3, 4]
print(list(it2))  # [0, 1, 2, 3, 4]

# Process differently
data = range(10)
evens_iter, odds_iter = itertools.tee(data, 2)

evens = [x for x in evens_iter if x % 2 == 0]
odds = [x for x in odds_iter if x % 2 == 1]

print(evens)  # [0, 2, 4, 6, 8]
print(odds)   # [1, 3, 5, 7, 9]

# Pairwise iteration
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)  # Advance b by one
    return zip(a, b)

numbers = [1, 2, 3, 4, 5]
for current, next_val in pairwise(numbers):
    print(f"{current} -> {next_val}")

# Moving average
def moving_average(iterable, n):
    iterators = itertools.tee(iterable, n)
    for i, it in enumerate(iterators):
        for _ in range(i):
            next(it, None)
    
    for values in zip(*iterators):
        yield sum(values) / n

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(moving_average(data, 3)))
# [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

# Warning: tee uses memory to store values
# Only use when iterators won't diverge too much

---

### Q60: What are comprehension alternatives for complex scenarios?

**Answer:**
When comprehensions become unreadable, use explicit loops, generator functions, or functional patterns.

```python
# Complex comprehension - hard to read
# Bad:
result = [
    process(x) if condition1(x) else alternative(x)
    for x in items
    if condition2(x) and condition3(x)
]

# Better: explicit loop
result = []
for x in items:
    if condition2(x) and condition3(x):
        if condition1(x):
            result.append(process(x))
        else:
            result.append(alternative(x))

# Or: generator function
def process_items(items):
    for x in items:
        if condition2(x) and condition3(x):
            if condition1(x):
                yield process(x)
            else:
                yield alternative(x)

result = list(process_items(items))

# Nested loops - can be confusing
# Comprehension:
result = [
    f(x, y)
    for x in range(10)
    if x % 2 == 0
    for y in range(10)
    if y % 3 == 0
]

# Better as explicit loops:
result = []
for x in range(10):
    if x % 2 == 0:
        for y in range(10):
            if y % 3 == 0:
                result.append(f(x, y))

# Use itertools for complex iterations
from itertools import product, combinations, permutations

# Instead of nested comprehensions:
pairs = [(x, y) for x in range(3) for y in range(3)]

# Use product:
pairs = list(product(range(3), repeat=2))

# Complex filtering - use filter + map
from functools import reduce

# Instead of complex comprehension:
result = [x ** 2 for x in range(100) if x % 3 == 0 if x % 5 == 0]

# Use filter + map:
result = list(map(
    lambda x: x ** 2,
    filter(lambda x: x % 3 == 0 and x % 5 == 0, range(100))
))

# Or generator function for clarity:
def filtered_squares():
    for x in range(100):
        if x % 3 == 0 and x % 5 == 0:
            yield x ** 2

result = list(filtered_squares())

---

### Q61: How do you implement efficient caching and memoization?

**Answer:**
Use `functools.lru_cache` for automatic memoization or implement custom caching strategies.

```python
from functools import lru_cache, cache

# Basic memoization with lru_cache
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))  # Fast!
print(fibonacci.cache_info())  # Cache statistics

# Unlimited cache (Python 3.9+)
@cache
def expensive_operation(x, y):
    print(f"Computing {x} + {y}")
    return x + y

print(expensive_operation(5, 3))  # "Computing 5 + 3", then 8
print(expensive_operation(5, 3))  # 8 (cached)

# Custom cache decorator
def memo(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memo
def slow_function(n):
    print(f"Computing for {n}")
    return n ** 2

print(slow_function(5))  # "Computing for 5", then 25
print(slow_function(5))  # 25 (cached)

# Cache with TTL (time-to-live)
import time

def timed_cache(seconds):
    def decorator(func):
        cache = {}
        cache_time = {}
        
        def wrapper(*args):
            now = time.time()
            if args in cache:
                if now - cache_time[args] < seconds:
                    return cache[args]
            
            result = func(*args)
            cache[args] = result
            cache_time[args] = now
            return result
        
        return wrapper
    return decorator

@timed_cache(seconds=5)
def get_data(query):
    print(f"Fetching {query}")
    return f"Data for {query}"

print(get_data("test"))  # "Fetching test"
print(get_data("test"))  # Cached
time.sleep(6)
print(get_data("test"))  # "Fetching test" again (cache expired)

# LRU cache with size limit
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)  # Mark as recently used
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove oldest

cache = LRUCache(3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
cache.put("d", 4)  # "a" evicted
print(cache.get("a"))  # None

# Property-level caching
from functools import cached_property

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    @cached_property
    def mean(self):
        print("Computing mean...")
        return sum(self.data) / len(self.data)

processor = DataProcessor([1, 2, 3, 4, 5])
print(processor.mean)  # "Computing mean...", then 3.0
print(processor.mean)  # 3.0 (cached)

# Conditional caching
def conditional_cache(condition_func):
    def decorator(func):
        cache = {}
        def wrapper(*args):
            if condition_func(*args):
                if args not in cache:
                    cache[args] = func(*args)
                return cache[args]
            return func(*args)
        return wrapper
    return decorator

@conditional_cache(lambda x: x > 10)
def process(x):
    print(f"Processing {x}")
    return x ** 2

print(process(5))   # "Processing 5" (not cached)
print(process(5))   # "Processing 5" again
print(process(15))  # "Processing 15" (cached)
print(process(15))  # Returned from cache

---

### Q62: How do you optimize memory usage with `__slots__`?

**Answer:**
`__slots__` restricts attributes and reduces memory overhead by preventing `__dict__` creation.

```python
# Without slots - uses __dict__
class PersonNormal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# With slots - more memory efficient
class PersonSlots:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Memory comparison
import sys

normal = PersonNormal("Alice", 30)
slotted = PersonSlots("Bob", 25)

print(f"Normal: {sys.getsizeof(normal) + sys.getsizeof(normal.__dict__)} bytes")
print(f"Slotted: {sys.getsizeof(slotted)} bytes")

# Slots prevent dynamic attribute addition
# normal.email = "alice@example.com"  # Works
# slotted.email = "bob@example.com"   # AttributeError!

# Slots with inheritance
class Person:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Employee(Person):
    __slots__ = ['employee_id']  # Add more slots
    
    def __init__(self, name, age, employee_id):
        super().__init__(name, age)
        self.employee_id = employee_id

emp = Employee("Charlie", 35, "E123")

# Slots with default values (Python 3.10+)
class Point:
    __slots__ = ('x', 'y')
    
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

# When to use slots:
# ✓ Creating millions of instances
# ✓ Memory-constrained environments
# ✓ Data classes with fixed attributes
# ✗ Need dynamic attributes
# ✗ Multiple inheritance scenarios
# ✗ When flexibility > performance

# Real-world example: Point class
class Point3D:
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def distance(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5

# Create million points - significant memory savings
points = [Point3D(i, i+1, i+2) for i in range(1000000)]

---

### Q63: What are memory views and how do they work?

**Answer:**
Memory views provide zero-copy access to array data, efficient for large binary data manipulation.

```python
# Without memoryview - creates copies
data = bytearray(b'Hello World')
slice1 = data[0:5]  # Creates copy
slice1[0] = ord('h')
print(data)  # b'Hello World' - original unchanged

# With memoryview - no copy
data = bytearray(b'Hello World')
view = memoryview(data)
slice_view = view[0:5]  # No copy!
slice_view[0] = ord('h')
print(data)  # b'hello World' - original changed!

# Performance with large data
import array

# Create large array
numbers = array.array('i', range(1000000))

# Memory efficient slicing
view = memoryview(numbers)
chunk = view[100:200]  # No copy created!

# Modify through memoryview
view[0] = 999
print(numbers[0])  # 999 - original modified

# Multi-dimensional data
import numpy as np  # If numpy available

# 2D array view
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# view = memoryview(arr)

# Working with different formats
data = bytearray(8)  # 8 bytes
view = memoryview(data)

# Cast to different type
int_view = view.cast('i')  # View as integers
int_view[0] = 12345

print(data.hex())  # Shows bytes

# Reading binary file efficiently
def read_large_binary(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        view = memoryview(data)
        # Process without copying
        for i in range(0, len(view), 1024):
            chunk = view[i:i+1024]
            process_chunk(chunk)

# Benefits:
# - No memory copies
# - Efficient slicing
# - Direct buffer access
# - Good for binary protocols

---

### Q64: How do you implement efficient data structures?

**Answer:**
Choose the right data structure and optimize access patterns for your use case.

```python
from collections import deque, defaultdict, Counter, OrderedDict
import heapq

# deque - O(1) append/pop from both ends
queue = deque()
queue.append(1)  # O(1) - right
queue.appendleft(0)  # O(1) - left
queue.pop()  # O(1) - right
queue.popleft()  # O(1) - left

# List is O(n) for left operations!
regular_list = []
regular_list.append(1)  # O(1)
regular_list.insert(0, 0)  # O(n) - slow!

# Use deque for queues
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    
    while queue:
        node = queue.popleft()  # O(1)
        if node not in visited:
            visited.add(node)
            queue.extend(graph[node])

# defaultdict - automatic default values
word_count = defaultdict(int)
for word in ["hello", "world", "hello"]:
    word_count[word] += 1  # No KeyError!

print(dict(word_count))  # {'hello': 2, 'world': 1}

# Grouping with defaultdict
from collections import defaultdict

students = [
    ('Alice', 'Math'),
    ('Bob', 'Science'),
    ('Charlie', 'Math')
]

by_subject = defaultdict(list)
for name, subject in students:
    by_subject[subject].append(name)

# Counter - counting made easy
from collections import Counter

text = "hello world hello python"
word_count = Counter(text.split())
print(word_count)  # Counter({'hello': 2, 'world': 1, 'python': 1})

# Most common
print(word_count.most_common(2))  # [('hello', 2), ('world', 1)]

# Heap (priority queue) - O(log n) operations
heap = []
heapq.heappush(heap, (3, 'task3'))  # (priority, task)
heapq.heappush(heap, (1, 'task1'))
heapq.heappush(heap, (2, 'task2'))

while heap:
    priority, task = heapq.heappop(heap)
    print(f"{task}: {priority}")
# task1: 1
# task2: 2
# task3: 3

# OrderedDict - remembers insertion order (dict is ordered in 3.7+)
from collections import OrderedDict

cache = OrderedDict()
cache['a'] = 1
cache['b'] = 2
cache['c'] = 3

# LRU cache behavior
cache.move_to_end('a')  # Move 'a' to end
cache.popitem(last=False)  # Remove oldest ('b')

# Set for O(1) lookups
large_list = list(range(1000000))
large_set = set(range(1000000))

# Slow
# 999999 in large_list  # O(n)

# Fast
999999 in large_set  # O(1)

# Bisect for sorted lists
import bisect

sorted_list = [1, 3, 5, 7, 9]
# Insert maintaining order
bisect.insort(sorted_list, 4)  # [1, 3, 4, 5, 7, 9]

# Find insertion point
pos = bisect.bisect_left(sorted_list, 5)  # 3

# Named tuples - memory efficient records
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # 10 20
print(p[0], p[1])  # Also works

# Performance comparison
import timeit

# List append vs deque append
list_time = timeit.timeit('l.append(1); l.pop(0)', 
                          setup='l=list(range(1000))', number=10000)
deque_time = timeit.timeit('d.append(1); d.popleft()', 
                           setup='from collections import deque; d=deque(range(1000))', 
                           number=10000)

print(f"List: {list_time:.4f}s")
print(f"Deque: {deque_time:.4f}s")
# Deque is much faster!

---

### Q65: How do you profile and optimize memory usage?

**Answer:**
Use memory profilers to identify bottlenecks and optimize data structures and algorithms.

```python
# Basic memory tracking
import sys

data = list(range(1000))
print(f"List size: {sys.getsizeof(data)} bytes")

# More accurate memory usage
def get_size(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(get_size(k) + get_size(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(get_size(item) for item in obj)
    return size

nested_data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
print(f"Total size: {get_size(nested_data)} bytes")

# Memory profiling with memory_profiler
# pip install memory-profiler
# @profile  # Uncomment when using memory_profiler
def memory_intensive():
    # Large list
    data = [i for i in range(1000000)]
    # Process
    result = [x ** 2 for x in data]
    return result

# Run with: python -m memory_profiler script.py

# Generator alternative - uses constant memory
def memory_efficient():
    data = range(1000000)  # Generator
    result = (x ** 2 for x in data)  # Generator
    return result

# Tracemalloc - built-in memory tracking
import tracemalloc

tracemalloc.start()

# Code to profile
data = [i for i in range(100000)]

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024:.2f} KB")
print(f"Peak: {peak / 1024:.2f} KB")

tracemalloc.stop()

# Find memory leaks
def snapshot_comparison():
    import tracemalloc
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Code that might leak
    leaked = []
    for i in range(10000):
        leaked.append([i] * 100)
    
    snapshot2 = tracemalloc.take_snapshot()
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    for stat in top_stats[:5]:
        print(stat)

# Object reference counting
import sys

data = [1, 2, 3]
print(sys.getrefcount(data))  # 2 (one from variable, one from argument)

# Weak references - don't prevent garbage collection
import weakref

class BigObject:
    pass

obj = BigObject()
weak_ref = weakref.ref(obj)

print(weak_ref())  # <BigObject object>
del obj
print(weak_ref())  # None - object was garbage collected

# Memory optimization tips:
# 1. Use generators instead of lists when possible
# 2. Use __slots__ for classes with many instances
# 3. Use array.array for homogeneous numeric data
# 4. Use sets for membership testing
# 5. Delete large objects explicitly when done
# 6. Use itertools for efficient iteration
# 7. Profile before optimizing!

---

## Section 5: Concurrency, Parallelism & Async Programming (Q66-80)

### Q66: What's the difference between concurrency and parallelism?

**Answer:**
Concurrency is about dealing with multiple tasks, parallelism is about doing multiple tasks simultaneously.

```python
# Concurrency: Multiple tasks in progress (not necessarily simultaneous)
# - Threading: Good for I/O-bound tasks
# - AsyncIO: Best for I/O-bound tasks
#
# Parallelism: Multiple tasks executed simultaneously
# - Multiprocessing: Good for CPU-bound tasks

# Example: Concurrent (one cook, multiple dishes)
import threading
import time

def cook_dish(dish_name):
    print(f"Start cooking {dish_name}")
    time.sleep(2)  # Simulate cooking
    print(f"Finished {dish_name}")

threads = []
for dish in ["pasta", "salad", "soup"]:
    thread = threading.Thread(target=cook_dish, args=(dish,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

# Example: Parallel (multiple cooks, each cooking own dish)
import multiprocessing

def process_data(chunk):
    return sum(x ** 2 for x in chunk)

if __name__ == '__main__':
    data = range(1000000)
    chunks = [range(i, i+250000) for i in range(0, 1000000, 250000)]
    
    with multiprocessing.Pool(4) as pool:
        results = pool.map(process_data, chunks)
    
    print(sum(results))

# Async: Concurrent without threads
import asyncio

async def fetch_data(url):
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # Simulate network request
    return f"Data from {url}"

async def main():
    urls = ["url1", "url2", "url3"]
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    print(results)

# asyncio.run(main())

# When to use what:
# - Threading: I/O-bound (file, network), light concurrency
# - AsyncIO: I/O-bound, many concurrent operations
# - Multiprocessing: CPU-bound, true parallelism needed

---

### Q67: How do you use threading in Python effectively?

**Answer:**
Threading enables concurrent execution but is limited by the GIL for CPU-bound tasks.

```python
import threading
import time

# Basic thread
def worker(name):
    print(f"Thread {name} starting")
    time.sleep(2)
    print(f"Thread {name} done")

thread = threading.Thread(target=worker, args=("A",))
thread.start()
thread.join()  # Wait for completion

# Multiple threads
threads = []
for i in range(5):
    thread = threading.Thread(target=worker, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

# Thread-safe counter with Lock
counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:  # Thread-safe
            counter += 1

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)  # 1000000 (correct with lock)

# Without lock - race condition!
# counter would be less than 1000000

# Thread pool for managing multiple threads
from concurrent.futures import ThreadPoolExecutor

def task(n):
    time.sleep(1)
    return n * 2

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(task, i) for i in range(10)]
    results = [f.result() for f in futures]

print(results)

# Map with thread pool
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(task, range(10)))

# Daemon threads - stop when main thread stops
def daemon_worker():
    while True:
        print("Daemon working...")
        time.sleep(1)

daemon = threading.Thread(target=daemon_worker, daemon=True)
daemon.start()
time.sleep(3)  # Daemon stops when program exits

# Thread-local storage
thread_local = threading.local()

def worker():
    thread_local.value = threading.current_thread().name
    print(f"Thread {thread_local.value}")

threads = [threading.Thread(target=worker) for _ in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Semaphore - limit concurrent access
semaphore = threading.Semaphore(3)  # Max 3 concurrent

def limited_access(n):
    with semaphore:
        print(f"Thread {n} accessing")
        time.sleep(1)
        print(f"Thread {n} done")

threads = [threading.Thread(target=limited_access, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Event for thread synchronization
event = threading.Event()

def waiter():
    print("Waiting for event...")
    event.wait()  # Block until set
    print("Event received!")

def setter():
    time.sleep(2)
    print("Setting event")
    event.set()

t1 = threading.Thread(target=waiter)
t2 = threading.Thread(target=setter)
t1.start()
t2.start()
t1.join()
t2.join()

---

### Q68: What is the Global Interpreter Lock (GIL) and how does it affect Python?

**Answer:**
The GIL is a mutex that prevents multiple threads from executing Python bytecode simultaneously.

```python
import threading
import time

# GIL Impact on CPU-bound tasks
def cpu_bound():
    total = 0
    for i in range(10_000_000):
        total += i
    return total

# Single thread
start = time.time()
result1 = cpu_bound()
result2 = cpu_bound()
print(f"Sequential: {time.time() - start:.2f}s")

# Multiple threads - NOT faster due to GIL!
start = time.time()
thread1 = threading.Thread(target=cpu_bound)
thread2 = threading.Thread(target=cpu_bound)
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print(f"Threading: {time.time() - start:.2f}s")  # Same or slower!

# Multiprocessing bypasses GIL
from multiprocessing import Process

start = time.time()
proc1 = Process(target=cpu_bound)
proc2 = Process(target=cpu_bound)
proc1.start()
proc2.start()
proc1.join()
proc2.join()
print(f"Multiprocessing: {time.time() - start:.2f}s")  # Faster!

# Threading is good for I/O-bound tasks
import requests  # Example

def io_bound(url):
    response = requests.get(url)
    return len(response.content)

# Threading works well here because waiting for I/O releases GIL
urls = ["https://example.com"] * 10

start = time.time()
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(io_bound, urls))
print(f"Threading I/O: {time.time() - start:.2f}s")  # Fast!

# GIL workarounds:
# 1. Multiprocessing for CPU-bound
# 2. AsyncIO for I/O-bound
# 3. C extensions (NumPy, etc) release GIL
# 4. Use different Python implementation (Jython, IronPython)

---

### Q69: How do you use multiprocessing for CPU-bound tasks?

**Answer:**
Multiprocessing creates separate Python processes, each with its own GIL, enabling true parallelism.

```python
from multiprocessing import Process, Pool, Queue, Manager, Value, Array
import time

# Basic process
def worker(name):
    print(f"Process {name} starting")
    time.sleep(1)
    print(f"Process {name} done")

if __name__ == '__main__':
    process = Process(target=worker, args=("A",))
    process.start()
    process.join()

# Process Pool - easier management
def square(x):
    return x ** 2

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        results = pool.map(square, range(10))
    print(results)

# Parallel processing of large dataset
def process_chunk(chunk):
    return sum(x ** 2 for x in chunk)

if __name__ == '__main__':
    data = range(1_000_000)
    chunk_size = 250_000
    chunks = [range(i, i+chunk_size) for i in range(0, 1_000_000, chunk_size)]
    
    with Pool(4) as pool:
        results = pool.map(process_chunk, chunks)
    
    total = sum(results)
    print(total)

# Shared memory between processes
from multiprocessing import Value, Array

def increment(shared_value, shared_array):
    shared_value.value += 1
    for i in range(len(shared_array)):
        shared_array[i] += 1

if __name__ == '__main__':
    shared_num = Value('i', 0)  # Shared integer
    shared_arr = Array('i', [0, 0, 0])  # Shared array
    
    processes = [Process(target=increment, args=(shared_num, shared_arr)) for _ in range(5)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(shared_num.value)  # 5
    print(list(shared_arr))  # [5, 5, 5]

# Queue for inter-process communication
from multiprocessing import Queue

def producer(queue):
    for i in range(5):
        queue.put(i)
    queue.put(None)  # Signal done

def consumer(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")

if __name__ == '__main__':
    q = Queue()
    prod = Process(target=producer, args=(q,))
    cons = Process(target=consumer, args=(q,))
    
    prod.start()
    cons.start()
    prod.join()
    cons.join()

# Manager for complex shared objects
from multiprocessing import Manager

def update_dict(shared_dict, key, value):
    shared_dict[key] = value

if __name__ == '__main__':
    with Manager() as manager:
        shared_dict = manager.dict()
        processes = []
        
        for i in range(5):
            p = Process(target=update_dict, args=(shared_dict, f'key{i}', i))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        print(dict(shared_dict))

# Process Pool with error handling
def risky_operation(x):
    if x == 5:
        raise ValueError("Bad value!")
    return x ** 2

if __name__ == '__main__':
    with Pool(4) as pool:
        try:
            results = pool.map(risky_operation, range(10))
        except ValueError as e:
            print(f"Error: {e}")

---

### Q70: How do you use asyncio for asynchronous programming?

**Answer:**
AsyncIO enables concurrent I/O operations using coroutines and an event loop.

```python
import asyncio

# Basic async function
async def say_hello():
    print("Hello")
    await asyncio.sleep(1)  # Non-blocking sleep
    print("World")

# Run async function
asyncio.run(say_hello())

# Multiple concurrent tasks
async def fetch_data(name):
    print(f"Fetching {name}")
    await asyncio.sleep(2)  # Simulate API call
    return f"Data from {name}"

async def main():
    # Run concurrently
    results = await asyncio.gather(
        fetch_data("API1"),
        fetch_data("API2"),
        fetch_data("API3")
    )
    print(results)

asyncio.run(main())

# Create tasks
async def main_with_tasks():
    task1 = asyncio.create_task(fetch_data("API1"))
    task2 = asyncio.create_task(fetch_data("API2"))
    
    # Do other work while tasks run
    print("Tasks started")
    
    # Wait for completion
    result1 = await task1
    result2 = await task2
    print(result1, result2)

asyncio.run(main_with_tasks())

# Async HTTP requests
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all():
    urls = [
        "https://example.com",
        "https://example.org",
        "https://example.net"
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

# asyncio.run(fetch_all())

# Async context manager
class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(1)
        return self
    
    async def __aexit__(self, *args):
        print("Releasing resource")
        await asyncio.sleep(1)

async def use_resource():
    async with AsyncResource() as resource:
        print("Using resource")

asyncio.run(use_resource())

# Async iterator
class AsyncCounter:
    def __init__(self, stop):
        self.current = 0
        self.stop = stop
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.current >= self.stop:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)
        self.current += 1
        return self.current

async def count():
    async for number in AsyncCounter(5):
        print(number)

asyncio.run(count())

# Timeout handling
async def slow_operation():
    await asyncio.sleep(10)
    return "Done"

async def with_timeout():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=2)
    except asyncio.TimeoutError:
        print("Operation timed out")

asyncio.run(with_timeout())

# Running sync code in async
import time

async def run_blocking():
    loop = asyncio.get_event_loop()
    # Run CPU-bound in thread pool
    result = await loop.run_in_executor(None, time.sleep, 2)
    return "Done"

---

### Q71: How do you handle race conditions and thread safety?

**Answer:**
Use locks, semaphores, and thread-safe data structures to prevent race conditions.

```python
import threading

# Race condition example - UNSAFE
counter = 0

def unsafe_increment():
    global counter
    for _ in range(100000):
        counter += 1  # NOT atomic!

threads = [threading.Thread(target=unsafe_increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)  # Less than 1000000 due to race condition!

# Thread-safe with Lock
counter = 0
lock = threading.Lock()

def safe_increment():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1

threads = [threading.Thread(target=safe_increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)  # 1000000 - correct!

# RLock - Reentrant Lock (can be acquired multiple times by same thread)
class BankAccount:
    def __init__(self):
        self.balance = 0
        self.lock = threading.RLock()
    
    def deposit(self, amount):
        with self.lock:
            self.balance += amount
    
    def withdraw(self, amount):
        with self.lock:
            if self.balance >= amount:
                self.balance -= amount
                return True
            return False
    
    def transfer(self, other, amount):
        with self.lock:  # Can acquire again
            if self.withdraw(amount):
                other.deposit(amount)

# Thread-safe queue
from queue import Queue, Empty

queue = Queue()

def producer():
    for i in range(5):
        queue.put(i)
        threading.Event().wait(0.1)

def consumer():
    while True:
        try:
            item = queue.get(timeout=1)
            print(f"Consumed: {item}")
            queue.task_done()
        except Empty:
            break

prod = threading.Thread(target=producer)
cons = threading.Thread(target=consumer)
prod.start()
cons.start()
prod.join()
cons.join()

# Condition variable for complex synchronization
condition = threading.Condition()
items = []

def consumer_cv():
    with condition:
        while not items:
            condition.wait()  # Wait for notification
        item = items.pop(0)
        print(f"Consumed: {item}")

def producer_cv():
    with condition:
        items.append(1)
        condition.notify()  # Wake up consumer

# Atomic operations with threading
from threading import Lock

class AtomicCounter:
    def __init__(self):
        self._value = 0
        self._lock = Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    def get(self):
        with self._lock:
            return self._value

---

### Q72: How do you use concurrent.futures for high-level concurrency?

**Answer:**
`concurrent.futures` provides a high-level interface for both threading and multiprocessing.

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time

# Thread pool
def task(n):
    time.sleep(1)
    return n * 2

with ThreadPoolExecutor(max_workers=5) as executor:
    # submit for individual tasks
    future = executor.submit(task, 5)
    result = future.result()
    print(result)  # 10
    
    # map for multiple tasks
    results = executor.map(task, range(10))
    print(list(results))

# Process pool for CPU-bound
def cpu_intensive(n):
    return sum(range(n))

with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(cpu_intensive, [1000000, 2000000, 3000000])
    print(list(results))

# as_completed - process results as they finish
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(task, i) for i in range(10)]
    
    for future in as_completed(futures):
        result = future.result()
        print(f"Got result: {result}")

# Error handling
def risky_task(n):
    if n == 5:
        raise ValueError("Bad value!")
    return n * 2

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(risky_task, i) for i in range(10)]
    
    for future in as_completed(futures):
        try:
            result = future.result()
            print(result)
        except ValueError as e:
            print(f"Error: {e}")

# Timeout handling
with ThreadPoolExecutor(max_workers=2) as executor:
    future = executor.submit(time.sleep, 10)
    try:
        result = future.result(timeout=2)
    except TimeoutError:
        print("Task timed out")

# Canceling futures
with ThreadPoolExecutor(max_workers=2) as executor:
    future = executor.submit(time.sleep, 10)
    time.sleep(1)
    cancelled = future.cancel()  # True if cancelled
    print(f"Cancelled: {cancelled}")

# Real-world example: web scraping
import requests

def fetch_url(url):
    response = requests.get(url)
    return (url, len(response.content))

urls = [f"https://example.com/page{i}" for i in range(10)]

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(fetch_url, urls))
    for url, size in results:
        print(f"{url}: {size} bytes")

---

### Q73: How do you implement producer-consumer patterns?

**Answer:**
Use queues for thread-safe producer-consumer communication.

```python
import threading
import queue
import time

# Basic producer-consumer with Queue
def producer(q, items):
    for item in items:
        print(f"Producing {item}")
        q.put(item)
        time.sleep(0.1)
    q.put(None)  # Sentinel value

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Consuming {item}")
        time.sleep(0.2)
        q.task_done()

q = queue.Queue()
prod = threading.Thread(target=producer, args=(q, range(5)))
cons = threading.Thread(target=consumer, args=(q,))

prod.start()
cons.start()
prod.join()
cons.join()

# Multiple producers and consumers
def multi_producer(q, name, items):
    for item in items:
        q.put(f"{name}-{item}")
    print(f"Producer {name} done")

def multi_consumer(q, name):
    while True:
        try:
            item = q.get(timeout=1)
            print(f"Consumer {name} got {item}")
            q.task_done()
        except queue.Empty:
            break

q = queue.Queue()

# Start producers
producers = [
    threading.Thread(target=multi_producer, args=(q, f"P{i}", range(3)))
    for i in range(2)
]

# Start consumers
consumers = [
    threading.Thread(target=multi_consumer, args=(q, f"C{i}"))
    for i in range(3)
]

for p in producers:
    p.start()
for c in consumers:
    c.start()

for p in producers:
    p.join()
q.join()  # Wait for all items to be processed

# Priority queue
priority_q = queue.PriorityQueue()

def priority_producer(q):
    q.put((1, "Low priority"))
    q.put((5, "High priority"))
    q.put((3, "Medium priority"))

def priority_consumer(q):
    while not q.empty():
        priority, item = q.get()
        print(f"Processing: {item} (priority: {priority})")

priority_producer(priority_q)
priority_consumer(priority_q)

# Bounded queue (max size)
bounded_q = queue.Queue(maxsize=3)

def bounded_producer(q):
    for i in range(10):
        q.put(i)  # Blocks if queue is full
        print(f"Produced {i}")

def bounded_consumer(q):
    while True:
        item = q.get()
        print(f"Consumed {item}")
        time.sleep(1)  # Slow consumer
        q.task_done()

---

### Q74: How do you use asyncio.Queue for async producer-consumer?

**Answer:**
AsyncIO provides async queues for coroutine-based producer-consumer patterns.

```python
import asyncio

async def producer(queue, n):
    for i in range(n):
        await asyncio.sleep(0.1)
        await queue.put(i)
        print(f"Produced {i}")
    await queue.put(None)  # Signal done

async def consumer(queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        await asyncio.sleep(0.2)
        print(f"Consumed {item}")

async def main():
    queue = asyncio.Queue()
    
    await asyncio.gather(
        producer(queue, 5),
        consumer(queue)
    )

asyncio.run(main())

# Multiple async producers/consumers
async def multi_producer(queue, name, n):
    for i in range(n):
        await queue.put(f"{name}-{i}")
        await asyncio.sleep(0.1)

async def multi_consumer(queue, name):
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=2)
            print(f"{name} consumed {item}")
        except asyncio.TimeoutError:
            break

async def main_multi():
    queue = asyncio.Queue()
    
    await asyncio.gather(
        multi_producer(queue, "P1", 3),
        multi_producer(queue, "P2", 3),
        multi_consumer(queue, "C1"),
        multi_consumer(queue, "C2")
    )

asyncio.run(main_multi())

---

### Q75: How do you handle deadlocks in Python?

**Answer:**
Prevent deadlocks through lock ordering, timeouts, and proper resource management.

```python
import threading
import time

# Deadlock example - TWO locks acquired in different order
lock1 = threading.Lock()
lock2 = threading.Lock()

def task1():
    with lock1:
        print("Task 1 has lock 1")
        time.sleep(0.1)
        with lock2:
            print("Task 1 has lock 2")

def task2():
    with lock2:  # Different order - DEADLOCK!
        print("Task 2 has lock 2")
        time.sleep(0.1)
        with lock1:
            print("Task 2 has lock 1")

# This will deadlock:
# t1 = threading.Thread(target=task1)
# t2 = threading.Thread(target=task2)
# t1.start()
# t2.start()

# Solution 1: Consistent lock ordering
def safe_task1():
    with lock1:
        with lock2:
            print("Safe task 1")

def safe_task2():
    with lock1:  # Same order
        with lock2:
            print("Safe task 2")

# Solution 2: Try-lock with timeout
def try_task():
    if lock1.acquire(timeout=1):
        try:
            if lock2.acquire(timeout=1):
                try:
                    print("Got both locks")
                finally:
                    lock2.release()
        finally:
            lock1.release()
    else:
        print("Couldn't acquire locks")

# Solution 3: Context manager with timeout
from contextlib import contextmanager

@contextmanager
def acquire_timeout(lock, timeout):
    result = lock.acquire(timeout=timeout)
    try:
        yield result
    finally:
        if result:
            lock.release()

def safe_with_timeout():
    with acquire_timeout(lock1, 1) as got_lock1:
        if got_lock1:
            with acquire_timeout(lock2, 1) as got_lock2:
                if got_lock2:
                    print("Success")

# Deadlock detection (simple)
import threading

class DeadlockDetector:
    def __init__(self):
        self.locks = {}
        self.lock = threading.Lock()
    
    def acquire(self, thread_id, lock_id):
        with self.lock:
            if thread_id not in self.locks:
                self.locks[thread_id] = []
            self.locks[thread_id].append(lock_id)
            
            # Check for circular wait
            if self.has_cycle():
                print("DEADLOCK DETECTED!")
                return False
            return True
    
    def has_cycle(self):
        # Simplified cycle detection
        # In real implementation, use proper graph algorithm
        return False

---

### Q76: How do you implement timeouts in concurrent code?

**Answer:**
Use timeout parameters and context managers to prevent indefinite blocking.

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Thread join with timeout
def long_running():
    time.sleep(10)

thread = threading.Thread(target=long_running)
thread.start()
thread.join(timeout=2)  # Wait max 2 seconds

if thread.is_alive():
    print("Thread still running after timeout")

# Lock with timeout
lock = threading.Lock()

def try_lock():
    if lock.acquire(timeout=2):
        try:
            print("Got lock")
        finally:
            lock.release()
    else:
        print("Lock timeout")

# Queue get with timeout
from queue import Queue, Empty

q = Queue()

try:
    item = q.get(timeout=1)
except Empty:
    print("Queue get timed out")

# Future with timeout
with ThreadPoolExecutor() as executor:
    future = executor.submit(time.sleep, 10)
    try:
        result = future.result(timeout=2)
    except TimeoutError:
        print("Future timed out")
        future.cancel()

# Asyncio timeout
import asyncio

async def slow_operation():
    await asyncio.sleep(10)

async def with_timeout():
    try:
        await asyncio.wait_for(slow_operation(), timeout=2)
    except asyncio.TimeoutError:
        print("Async operation timed out")

asyncio.run(with_timeout())

# Context manager for timeout
import signal
from contextlib import contextmanager

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Use:
# try:
#     with time_limit(5):
#         long_running_task()
# except TimeoutError:
#     print("Task timed out")

---

### Q77: How do you use semaphores and barriers?

**Answer:**
Semaphores limit concurrent access, barriers synchronize multiple threads.

```python
import threading
import time

# Semaphore - limit concurrent access
semaphore = threading.Semaphore(3)  # Max 3 concurrent

def limited_resource(n):
    with semaphore:
        print(f"Thread {n} accessing")
        time.sleep(2)
        print(f"Thread {n} done")

threads = [threading.Thread(target=limited_resource, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
# Only 3 threads run simultaneously

# BoundedSemaphore - prevents release() above initial value
bounded = threading.BoundedSemaphore(2)

# Barrier - wait for all threads
barrier = threading.Barrier(3)  # Wait for 3 threads

def worker(n):
    print(f"Thread {n} starting")
    time.sleep(n)
    print(f"Thread {n} waiting at barrier")
    barrier.wait()  # All threads wait here
    print(f"Thread {n} passed barrier")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Real example: rate limiting
class RateLimiter:
    def __init__(self, max_calls, time_window):
        self.semaphore = threading.Semaphore(max_calls)
        self.time_window = time_window
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.semaphore.acquire()
            try:
                result = func(*args, **kwargs)
                # Release after time window
                threading.Timer(self.time_window, self.semaphore.release).start()
                return result
            except:
                self.semaphore.release()
                raise
        return wrapper

@RateLimiter(max_calls=5, time_window=1)
def api_call():
    print("API call made")

---

### Q78: How do you implement async context managers?

**Answer:**
Async context managers use `async with` for async setup/teardown.

```python
import asyncio

# Basic async context manager
class AsyncDatabase:
    async def __aenter__(self):
        print("Connecting to database")
        await asyncio.sleep(1)
        self.connection = "DB Connection"
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Closing database")
        await asyncio.sleep(1)
        return False

async def use_db():
    async with AsyncDatabase() as db:
        print(f"Using {db.connection}")

asyncio.run(use_db())

# Async context manager with asynccontextmanager
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_timer(label):
    import time
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{label}: {end - start:.2f}s")

async def timed_operation():
    async with async_timer("Operation"):
        await asyncio.sleep(2)
        print("Working...")

asyncio.run(timed_operation())

# File handling async
@asynccontextmanager
async def async_open(filename, mode='r'):
    # In real code, use aiofiles library
    file = open(filename, mode)
    try:
        yield file
    finally:
        file.close()

# HTTP session management
import aiohttp

@asynccontextmanager
async def http_session():
    session = aiohttp.ClientSession()
    try:
        yield session
    finally:
        await session.close()

async def fetch_data():
    async with http_session() as session:
        async with session.get('https://example.com') as response:
            return await response.text()

---

### Q79: How do you handle signals in async code?

**Answer:**
Use asyncio's signal handling for graceful shutdown and interruption.

```python
import asyncio
import signal

# Basic signal handling
async def main():
    loop = asyncio.get_running_loop()
    
    def handle_signal(sig):
        print(f"Received signal {sig}")
        loop.stop()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
    
    try:
        while True:
            await asyncio.sleep(1)
            print("Working...")
    except KeyboardInterrupt:
        print("Interrupted")

# asyncio.run(main())

# Graceful shutdown
class Application:
    def __init__(self):
        self.running = True
        self.tasks = []
    
    async def worker(self, name):
        while self.running:
            print(f"{name} working")
            await asyncio.sleep(1)
        print(f"{name} shutting down")
    
    def shutdown(self):
        print("Shutting down...")
        self.running = False
    
    async def run(self):
        loop = asyncio.get_running_loop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.shutdown)
        
        self.tasks = [
            asyncio.create_task(self.worker(f"Worker-{i}"))
            for i in range(3)
        ]
        
        await asyncio.gather(*self.tasks)

# app = Application()
# asyncio.run(app.run())

---

### Q80: How do you debug concurrent Python code?

**Answer:**
Use logging, threading utilities, and specialized tools to debug concurrent issues.

```python
import threading
import logging
import time

# Configure logging for threads
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(threadName)s - %(message)s'
)

def worker():
    logging.debug("Starting")
    time.sleep(1)
    logging.debug("Finished")

threads = [threading.Thread(target=worker, name=f"Worker-{i}") for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Thread enumeration
def show_threads():
    for thread in threading.enumerate():
        print(f"Thread: {thread.name}, Alive: {thread.is_alive()}")

# Lock debugging
class DebugLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.owner = None
    
    def acquire(self, blocking=True):
        thread = threading.current_thread().name
        logging.debug(f"{thread} trying to acquire lock")
        result = self.lock.acquire(blocking)
        if result:
            self.owner = thread
            logging.debug(f"{thread} acquired lock")
        return result
    
    def release(self):
        thread = threading.current_thread().name
        logging.debug(f"{thread} releasing lock")
        self.lock.release()
        self.owner = None

# Trace concurrent execution
import sys

def trace_calls(frame, event, arg):
    if event == 'call':
        code = frame.f_code
        print(f"{threading.current_thread().name}: {code.co_filename}:{code.co_name}")
    return trace_calls

# sys.settrace(trace_calls)

# Deadlock detection
import threading
from collections import defaultdict

class DeadlockDetector:
    def __init__(self):
        self.waiting_for = defaultdict(set)
        self.held_locks = defaultdict(set)
        self.lock = threading.Lock()
    
    def acquire(self, thread_id, lock_id):
        with self.lock:
            self.waiting_for[thread_id].add(lock_id)
            if self.would_deadlock(thread_id, lock_id):
                return False
            self.held_locks[thread_id].add(lock_id)
            self.waiting_for[thread_id].remove(lock_id)
            return True
    
    def would_deadlock(self, thread_id, lock_id):
        # Check if acquiring would create cycle
        # Simplified - real implementation needs graph traversal
        for other_thread, locks in self.held_locks.items():
            if lock_id in locks:
                if other_thread in self.waiting_for[thread_id]:
                    return True
        return False

# Performance profiling
import cProfile
import pstats

def profile_concurrent():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Concurrent code here
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()

# Tips:
# 1. Use logging with thread names
# 2. Add assertions for invariants
# 3. Use thread-safe data structures
# 4. Test with different thread counts
# 5. Use race condition detectors (ThreadSanitizer)
# 6. Enable Python's development mode: python -X dev

---

## Section 6: Performance Optimization & Profiling (Q81-95)

### Q81: How do you profile Python code to find bottlenecks?

**Answer:**
Use profilers to identify slow functions and optimize based on data, not assumptions.

```python
# cProfile - built-in profiler
import cProfile
import pstats

def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

# Profile with cProfile
profiler = cProfile.Profile()
profiler.enable()

slow_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions

# Command line profiling
# python -m cProfile -s cumulative script.py

# Line profiler - line-by-line timing
# pip install line_profiler
# @profile  # Uncomment when using kernprof
def detailed_function():
    numbers = []
    for i in range(1000):
        numbers.append(i ** 2)
    return sum(numbers)

# Run with: kernprof -l -v script.py

# Timer decorator
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def my_function():
    time.sleep(1)
    return "Done"

my_function()

# Timing specific code blocks
import time

start = time.perf_counter()
# Code to measure
result = sum(range(1000000))
end = time.perf_counter()
print(f"Elapsed: {end - start:.4f}s")

# timeit for accurate micro-benchmarks
import timeit

# Time a single statement
time_taken = timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
print(f"Time: {time_taken:.4f}s")

# Compare implementations
def method1():
    return [x**2 for x in range(1000)]

def method2():
    return list(map(lambda x: x**2, range(1000)))

time1 = timeit.timeit(method1, number=1000)
time2 = timeit.timeit(method2, number=1000)

print(f"Method 1: {time1:.4f}s")
print(f"Method 2: {time2:.4f}s")

# Memory profiling
# pip install memory-profiler
# @profile
def memory_intensive():
    big_list = [i for i in range(1000000)]
    return len(big_list)

# Run with: python -m memory_profiler script.py

---

### Q82: What are Python's built-in performance optimization techniques?

**Answer:**
Use list comprehensions, generators, built-in functions, and appropriate data structures.

```python
# List comprehension vs loop - comprehension is faster
# Slow
result = []
for i in range(1000):
    result.append(i ** 2)

# Fast
result = [i ** 2 for i in range(1000)]

# Generator for memory efficiency
# Memory heavy
big_list = [x ** 2 for x in range(1000000)]

# Memory efficient
big_gen = (x ** 2 for x in range(1000000))

# Use built-in functions - they're optimized in C
# Slow
total = 0
for i in range(1000):
    total += i

# Fast
total = sum(range(1000))

# Local variable lookup is faster
import math

# Slow - global lookup each iteration
def slow():
    for i in range(1000):
        x = math.sqrt(i)

# Fast - local variable
def fast():
    sqrt = math.sqrt
    for i in range(1000):
        x = sqrt(i)

# Use set for membership testing
# Slow
if item in large_list:  # O(n)
    pass

# Fast
if item in large_set:  # O(1)
    pass

# String concatenation
# Slow
result = ""
for s in strings:
    result += s  # Creates new string each time!

# Fast
result = "".join(strings)

# Dict.get() vs try/except
# Fast for common case
value = d.get(key, default)

# Fast when exceptions are rare
try:
    value = d[key]
except KeyError:
    value = default

# Use __slots__ for classes with many instances
class Point:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Faster than dict-based attributes

---

### Q83: How do you optimize loops and iterations?

**Answer:**
Move invariants out of loops, use appropriate iteration methods, and consider vectorization.

```python
# Move calculations out of loop
# Slow
for i in range(1000):
    result = expensive_function() * i  # Called 1000 times!

# Fast
temp = expensive_function()  # Called once
for i in range(1000):
    result = temp * i

# Use enumerate instead of range(len())
# Slow
for i in range(len(items)):
    print(i, items[i])

# Fast
for i, item in enumerate(items):
    print(i, item)

# Use zip for parallel iteration
# Slow
for i in range(len(list1)):
    process(list1[i], list2[i])

# Fast
for item1, item2 in zip(list1, list2):
    process(item1, item2)

# List comprehension vs map + lambda
# Generally faster
result = [x * 2 for x in numbers]

# Sometimes slower due to lambda overhead
result = list(map(lambda x: x * 2, numbers))

# But fast with built-in function
result = list(map(str, numbers))

# Use itertools for efficient iteration
from itertools import chain, islice

# Flatten lists
nested = [[1, 2], [3, 4], [5, 6]]
flat = list(chain.from_iterable(nested))  # Fast

# vs slow manual approach
flat = []
for sublist in nested:
    flat.extend(sublist)

# Avoid repeated attribute lookup in loops
# Slow
for i in range(1000):
    my_object.method()

# Fast
method = my_object.method
for i in range(1000):
    method()

---

### Q84: How do you optimize memory usage?

**Answer:**
Use generators, slots, appropriate data structures, and manage object lifecycle.

```python
# Generators vs lists
# Memory heavy
def get_numbers_list(n):
    return [i for i in range(n)]

numbers = get_numbers_list(1000000)  # Lots of memory

# Memory efficient
def get_numbers_gen(n):
    for i in range(n):
        yield i

numbers = get_numbers_gen(1000000)  # Tiny memory

# __slots__ for attribute storage
# Heavy
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Efficient
class PersonSlots:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Use array for homogeneous data
from array import array

# Heavy for numeric data
numbers_list = [1, 2, 3, 4, 5] * 1000

# Efficient
numbers_array = array('i', [1, 2, 3, 4, 5] * 1000)

# Lazy evaluation
class LazyProperty:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
    
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = self.func(obj)
        setattr(obj, self.name, value)
        return value

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    @LazyProperty
    def expensive_result(self):
        return sum(x ** 2 for x in self.data)

# Delete large objects when done
large_data = [i for i in range(1000000)]
process(large_data)
del large_data  # Free memory immediately

# Use context managers for resource cleanup
with open('file.txt') as f:
    data = f.read()
# File automatically closed

# Avoid circular references
import weakref

class Node:
    def __init__(self, value):
        self.value = value
        self.parent = None  # Regular reference
    
    def set_parent(self, parent):
        self.parent = weakref.ref(parent)  # Weak reference

---

### Q85: How do you use NumPy for performance optimization?

**Answer:**
NumPy provides vectorized operations that are much faster than Python loops.

```python
import numpy as np
import time

# Python list - slow
python_list = list(range(1000000))
start = time.time()
result = [x ** 2 for x in python_list]
print(f"Python: {time.time() - start:.4f}s")

# NumPy array - fast
numpy_array = np.arange(1000000)
start = time.time()
result = numpy_array ** 2
print(f"NumPy: {time.time() - start:.4f}s")
# NumPy is 10-100x faster!

# Vectorized operations
# Slow - Python loop
def python_distance(x1, y1, x2, y2):
    return [(x2[i] - x1[i])**2 + (y2[i] - y1[i])**2 for i in range(len(x1))]

# Fast - NumPy vectorization
def numpy_distance(x1, y1, x2, y2):
    return (x2 - x1)**2 + (y2 - y1)**2

x1 = np.random.rand(1000000)
y1 = np.random.rand(1000000)
x2 = np.random.rand(1000000)
y2 = np.random.rand(1000000)

start = time.time()
result = numpy_distance(x1, y1, x2, y2)
print(f"NumPy: {time.time() - start:.4f}s")

# Matrix operations
matrix1 = np.random.rand(1000, 1000)
matrix2 = np.random.rand(1000, 1000)

# Fast matrix multiplication
result = np.dot(matrix1, matrix2)

# Broadcasting
array = np.array([1, 2, 3, 4, 5])
result = array + 10  # Adds 10 to each element

# Boolean indexing - fast filtering
array = np.arange(1000)
result = array[array > 500]  # Much faster than list comprehension

# In-place operations save memory
array = np.arange(1000000)
array += 1  # In-place, no new array created

---

## Section 7: Testing, Debugging & Code Quality (Q86-95)

---

### Q86: How do you optimize database queries in Python?

**Answer:**
Use connection pooling, prepared statements, batch operations, and indexing.

```python
import sqlite3
from contextlib import contextmanager

# Connection pooling
class DatabasePool:
    def __init__(self, db_path, pool_size=5):
        self.db_path = db_path
        self.pool = [sqlite3.connect(db_path) for _ in range(pool_size)]
        self.available = self.pool.copy()
    
    @contextmanager
    def get_connection(self):
        conn = self.available.pop()
        try:
            yield conn
        finally:
            self.available.append(conn)

# Batch operations instead of individual inserts
# Slow
def slow_insert(conn, items):
    for item in items:
        conn.execute("INSERT INTO items VALUES (?)", (item,))
    conn.commit()

# Fast
def fast_insert(conn, items):
    conn.executemany("INSERT INTO items VALUES (?)", [(item,) for item in items])
    conn.commit()

# Use prepared statements
# Slow - SQL injection risk and slower
def unsafe_query(conn, user_id):
    result = conn.execute(f"SELECT * FROM users WHERE id = {user_id}")
    return result.fetchall()

# Fast and safe
def safe_query(conn, user_id):
    result = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,))
    return result.fetchall()

# Fetch only what you need
# Slow - fetches all columns
def fetch_all_columns(conn):
    return conn.execute("SELECT * FROM users").fetchall()

# Fast - fetch specific columns
def fetch_needed_columns(conn):
    return conn.execute("SELECT id, name FROM users").fetchall()

# Use fetchmany for large results
def process_large_result(conn):
    cursor = conn.execute("SELECT * FROM large_table")
    while True:
        rows = cursor.fetchmany(1000)  # Fetch in batches
        if not rows:
            break
        process_batch(rows)

# Index optimization
def create_indexes(conn):
    conn.execute("CREATE INDEX IF NOT EXISTS idx_user_email ON users(email)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_order_date ON orders(date)")

# Connection reuse
@contextmanager
def get_db_connection():
    conn = sqlite3.connect('database.db')
    conn.row_factory = sqlite3.Row  # Dict-like access
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

---

### Q87: How do you optimize string operations?

**Answer:**
Use appropriate methods, avoid concatenation in loops, and use f-strings.

```python
# String concatenation - avoid in loops
# Slow
result = ""
for i in range(10000):
    result += str(i)  # Creates new string each time!

# Fast
result = "".join(str(i) for i in range(10000))

# String formatting
name = "Alice"
age = 30

# Slow (old style)
msg = "Name: %s, Age: %d" % (name, age)

# Good
msg = "Name: {}, Age: {}".format(name, age)

# Best (Python 3.6+)
msg = f"Name: {name}, Age: {age}"

# String searching
text = "hello world " * 1000

# Slow for multiple searches
if "hello" in text and "world" in text:
    pass

# Fast - compile regex for repeated use
import re
pattern = re.compile(r'hello|world')
if pattern.search(text):
    pass

# String splitting
# Good for simple splits
parts = "a,b,c,d".split(",")

# Better for complex patterns
import re
parts = re.split(r'[,;:]', "a,b;c:d")

# Use str methods instead of regex when possible
# Slow
import re
if re.match(r'^hello', text):
    pass

# Fast
if text.startswith('hello'):
    pass

# String building with list
# Fast for many strings
parts = []
for i in range(1000):
    parts.append(str(i))
result = "".join(parts)

# Bytes vs strings
# Use bytes for binary data
data = b"Hello"  # Bytes
text = "Hello"   # String

# Converting
text_to_bytes = text.encode('utf-8')
bytes_to_text = data.decode('utf-8')

---

### Q88: How do you use caching effectively?

**Answer:**
Implement smart caching strategies with appropriate eviction policies and TTL.

```python
from functools import lru_cache, cache
import time

# Basic memoization
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Unlimited cache (Python 3.9+)
@cache
def expensive_function(x):
    time.sleep(1)
    return x ** 2

# Custom cache with TTL
class TTLCache:
    def __init__(self, ttl=60):
        self.cache = {}
        self.ttl = ttl
    
    def get(self, key):
        if key in self.cache:
            value, timestamp = self.cache[key]
            if time.time() - timestamp < self.ttl:
                return value
            else:
                del self.cache[key]
        return None
    
    def set(self, key, value):
        self.cache[key] = (value, time.time())

# LRU Cache implementation
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)

# Decorator with custom cache
def smart_cache(maxsize=128, ttl=None):
    def decorator(func):
        cache = {}
        timestamps = {}
        
        def wrapper(*args):
            # Check TTL
            if ttl and args in timestamps:
                if time.time() - timestamps[args] > ttl:
                    del cache[args]
                    del timestamps[args]
            
            # Return cached or compute
            if args not in cache:
                cache[args] = func(*args)
                if ttl:
                    timestamps[args] = time.time()
            
            # LRU eviction
            if len(cache) > maxsize:
                oldest = min(timestamps.items(), key=lambda x: x[1])[0]
                del cache[oldest]
                del timestamps[oldest]
            
            return cache[args]
        
        return wrapper
    return decorator

@smart_cache(maxsize=100, ttl=60)
def cached_api_call(endpoint):
    # Simulate API call
    return f"Data from {endpoint}"

---

### Q89: How do you optimize file I/O operations?

**Answer:**
Use buffering, binary mode, and appropriate read/write strategies.

```python
# Buffered reading
# Slow - read one byte at a time
with open('file.txt', 'r') as f:
    while True:
        char = f.read(1)
        if not char:
            break
        process(char)

# Fast - use buffering
with open('file.txt', 'r') as f:
    while True:
        chunk = f.read(8192)  # Read 8KB at a time
        if not chunk:
            break
        process(chunk)

# Line-by-line for text files
# Good - memory efficient
with open('file.txt', 'r') as f:
    for line in f:  # Uses buffering automatically
        process(line)

# Binary mode for non-text files
# Fast for binary data
with open('data.bin', 'rb') as f:
    data = f.read()

# Writing efficiently
# Slow - multiple writes
with open('output.txt', 'w') as f:
    for item in items:
        f.write(str(item) + '\n')

# Fast - batch writes
with open('output.txt', 'w') as f:
    f.write('\n'.join(str(item) for item in items))

# Or use writelines
with open('output.txt', 'w') as f:
    f.writelines(f"{item}\n" for item in items)

# Memory-mapped files for large files
import mmap

with open('large_file.bin', 'r+b') as f:
    mmapped = mmap.mmap(f.fileno(), 0)
    # Access like array
    data = mmapped[1000:2000]
    mmapped.close()

# Use pathlib for path operations
from pathlib import Path

# Modern and cross-platform
path = Path('data/file.txt')
if path.exists():
    content = path.read_text()

---

### Q90: How do you use Python's C extensions for performance?

**Answer:**
Use Cython, ctypes, or write C extensions for performance-critical code.

```python
# Cython example (save as example.pyx)
# def fibonacci_cy(int n):
#     if n < 2:
#         return n
#     return fibonacci_cy(n-1) + fibonacci_cy(n-2)

# Compile: cythonize -i example.pyx

# Using NumPy (C-optimized)
import numpy as np

def pure_python_sum(data):
    return sum(data)

def numpy_sum(data):
    return np.sum(data)

# NumPy is much faster for large arrays

# ctypes for calling C libraries
from ctypes import *

# Load C library
# libc = CDLL('libc.so.6')  # Linux
# result = libc.printf(b"Hello from C!\n")

# Using array module (C-optimized)
from array import array

# Fast for homogeneous numeric data
numbers = array('i', range(1000000))

# Using struct for binary data
import struct

# Pack data to bytes
data = struct.pack('iii', 1, 2, 3)

# Unpack bytes to data
values = struct.unpack('iii', data)

---

### Q91: How do you optimize recursive algorithms?

**Answer:**
Use memoization, iterative alternatives, or tail recursion patterns.

```python
# Naive recursion - slow
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# Memoized recursion - fast
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_memo(n):
    if n < 2:
        return n
    return fibonacci_memo(n-1) + fibonacci_memo(n-2)

# Iterative - fastest
def fibonacci_iter(n):
    if n < 2:
        return n
    a, b = 0, 1
    for _ in range(n-1):
        a, b = b, a + b
    return b

# Tail recursion (Python doesn't optimize, but pattern is useful)
def factorial_tail(n, accumulator=1):
    if n <= 1:
        return accumulator
    return factorial_tail(n-1, n * accumulator)

# Convert to iterative
def factorial_iter(n):
    result = 1
    for i in range(2, n+1):
        result *= i
    return result

# Memoization decorator
def memoize(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def expensive_recursive(n):
    if n <= 1:
        return 1
    return expensive_recursive(n-1) + expensive_recursive(n-2)

---

### Q92: How do you optimize imports and module loading?

**Answer:**
Import only what you need, use lazy imports, and understand import costs.

```python
# Bad - imports everything
from module import *

# Good - import only what's needed
from module import specific_function

# Lazy imports for optional dependencies
def process_data():
    import pandas as pd  # Only imported when function is called
    return pd.DataFrame()

# Import at module level for frequently used
import json  # Good - at top

def process():
    # import json  # Bad - imports every time function runs
    return json.loads(data)

# Conditional imports
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

if HAS_NUMPY:
    # Use NumPy
    pass
else:
    # Fallback
    pass

# Import aliasing for long names
import some_long_module_name as short

# Local imports for speed
def my_function():
    from math import sqrt  # Faster lookup
    return sqrt(42)

# Avoid circular imports
# file1.py
# from file2 import func2  # Bad if file2 imports from file1

# Better: import at function level
def func1():
    from file2 import func2
    return func2()

---

### Q93: How do you benchmark and compare implementations?

**Answer:**
Use timeit for accurate benchmarking and compare multiple approaches.

```python
import timeit
import time

# Basic timing
start = time.perf_counter()
result = expensive_operation()
elapsed = time.perf_counter() - start

# timeit for accurate micro-benchmarks
# Runs multiple times and takes average
time1 = timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)

# Compare multiple implementations
def method1():
    return [x**2 for x in range(1000)]

def method2():
    return list(map(lambda x: x**2, range(1000)))

def method3():
    result = []
    for x in range(1000):
        result.append(x**2)
    return result

# Benchmark
print(f"Method 1: {timeit.timeit(method1, number=1000):.4f}s")
print(f"Method 2: {timeit.timeit(method2, number=1000):.4f}s")
print(f"Method 3: {timeit.timeit(method3, number=1000):.4f}s")

# Setup and teardown
setup = "from math import sqrt"
stmt = "sqrt(144)"
time_taken = timeit.timeit(stmt, setup=setup, number=100000)

# Benchmark decorator
def benchmark(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__}: {end - start:.6f}s")
        return result
    return wrapper

@benchmark
def slow_function():
    time.sleep(0.1)

# Compare with context manager
from contextlib import contextmanager

@contextmanager
def timer(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label}: {end - start:.4f}s")

with timer("Operation"):
    result = expensive_operation()

---

### Q94: How do you optimize dictionary and set operations?

**Answer:**
Use appropriate methods, understand hash operations, and choose right data structures.

```python
# Dictionary initialization
# Slow
d = {}
for i in range(1000):
    d[i] = i ** 2

# Fast
d = {i: i**2 for i in range(1000)}

# Or use dict()
keys = range(1000)
values = [i**2 for i in range(1000)]
d = dict(zip(keys, values))

# Membership testing
# Slow - list
if item in large_list:  # O(n)
    pass

# Fast - set
if item in large_set:  # O(1)
    pass

# Removing duplicates
# Slow
unique = []
for item in items:
    if item not in unique:
        unique.append(item)

# Fast
unique = list(set(items))

# Dictionary merging (Python 3.9+)
d1 = {'a': 1, 'b': 2}
d2 = {'b': 3, 'c': 4}

# Old way
merged = {**d1, **d2}

# New way (Python 3.9+)
merged = d1 | d2

# Dictionary comprehension with filtering
# Fast
result = {k: v for k, v in large_dict.items() if v > 10}

# Get with default
# Good
value = d.get(key, default)

# Or use defaultdict
from collections import defaultdict
d = defaultdict(int)
d['key'] += 1  # No KeyError

# Set operations
set1 = {1, 2, 3, 4, 5}
set2 = {4, 5, 6, 7, 8}

# Fast operations
intersection = set1 & set2  # {4, 5}
union = set1 | set2  # {1, 2, 3, 4, 5, 6, 7, 8}
difference = set1 - set2  # {1, 2, 3}
symmetric_diff = set1 ^ set2  # {1, 2, 3, 6, 7, 8}

---

### Q95: How do you optimize class instantiation?

**Answer:**
Use __slots__, object pooling, and flyweight pattern for many instances.

```python
# __slots__ for memory efficiency
class Point:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Object pooling
class ObjectPool:
    def __init__(self, cls, size=10):
        self.cls = cls
        self.pool = [cls() for _ in range(size)]
        self.available = self.pool.copy()
    
    def acquire(self):
        if self.available:
            return self.available.pop()
        return self.cls()  # Create new if pool empty
    
    def release(self, obj):
        self.available.append(obj)

# Use pool
pool = ObjectPool(Point, size=100)
p = pool.acquire()
# Use point
pool.release(p)

# Flyweight pattern for shared state
class Flyweight:
    _instances = {}
    
    def __new__(cls, shared_state):
        if shared_state not in cls._instances:
            instance = super().__new__(cls)
            cls._instances[shared_state] = instance
        return cls._instances[shared_state]

# Lazy initialization
class LazyClass:
    def __init__(self):
        self._expensive = None
    
    @property
    def expensive(self):
        if self._expensive is None:
            self._expensive = expensive_computation()
        return self._expensive

# Factory with caching
class CachedFactory:
    _cache = {}
    
    @classmethod
    def create(cls, key):
        if key not in cls._cache:
            cls._cache[key] = ExpensiveObject(key)
        return cls._cache[key]

---

## Section 7: Testing, Debugging & Code Quality (Q96-110)

### Q96: How do you write effective unit tests in Python?

**Answer:**
Use unittest or pytest with proper test structure, mocking, and assertions.

```python
import unittest
from unittest.mock import Mock, patch, MagicMock

# Basic unittest
class TestCalculator(unittest.TestCase):
    def setUp(self):
        """Run before each test"""
        self.calc = Calculator()
    
    def tearDown(self):
        """Run after each test"""
        pass
    
    def test_add(self):
        result = self.calc.add(2, 3)
        self.assertEqual(result, 5)
    
    def test_divide_by_zero(self):
        with self.assertRaises(ZeroDivisionError):
            self.calc.divide(10, 0)
    
    def test_multiple_assertions(self):
        self.assertTrue(True)
        self.assertFalse(False)
        self.assertIsNone(None)
        self.assertIn(1, [1, 2, 3])

# pytest - simpler syntax
def test_add():
    calc = Calculator()
    assert calc.add(2, 3) == 5

def test_divide_by_zero():
    calc = Calculator()
    with pytest.raises(ZeroDivisionError):
        calc.divide(10, 0)

# Fixtures in pytest
import pytest

@pytest.fixture
def calculator():
    return Calculator()

def test_with_fixture(calculator):
    assert calculator.add(2, 3) == 5

# Mocking external dependencies
def test_api_call():
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {'data': 'test'}
        result = fetch_data('https://api.example.com')
        assert result == {'data': 'test'}

# Mock object
def test_with_mock():
    mock_db = Mock()
    mock_db.query.return_value = [1, 2, 3]
    
    result = process_data(mock_db)
    mock_db.query.assert_called_once()

# Parametrized tests
@pytest.mark.parametrize("a,b,expected", [
    (2, 3, 5),
    (0, 0, 0),
    (-1, 1, 0),
    (100, 200, 300)
])
def test_add_parametrized(a, b, expected):
    calc = Calculator()
    assert calc.add(a, b) == expected

# Test coverage
# Run: pytest --cov=mymodule tests/

---

### Q97: How do you implement test-driven development (TDD)?

**Answer:**
Write failing tests first, implement minimal code to pass, then refactor.

```python
# TDD Cycle: Red -> Green -> Refactor

# Step 1: Write failing test (RED)
def test_calculate_total():
    cart = ShoppingCart()
    cart.add_item("Book", 10.00)
    cart.add_item("Pen", 2.50)
    assert cart.total() == 12.50

# Step 2: Implement minimal code (GREEN)
class ShoppingCart:
    def __init__(self):
        self.items = []
    
    def add_item(self, name, price):
        self.items.append({'name': name, 'price': price})
    
    def total(self):
        return sum(item['price'] for item in self.items)

# Step 3: Refactor
class ShoppingCart:
    def __init__(self):
        self._items = []
    
    def add_item(self, name, price):
        if price < 0:
            raise ValueError("Price cannot be negative")
        self._items.append({'name': name, 'price': price})
    
    def total(self):
        return round(sum(item['price'] for item in self._items), 2)
    
    @property
    def item_count(self):
        return len(self._items)

# Add more tests
def test_empty_cart():
    cart = ShoppingCart()
    assert cart.total() == 0

def test_negative_price():
    cart = ShoppingCart()
    with pytest.raises(ValueError):
        cart.add_item("Book", -10)

---

### Q98: How do you debug Python code effectively?

**Answer:**
Use debugger, logging, print statements, and IDE debugging tools.

```python
# pdb - Python debugger
import pdb

def complex_function(x):
    result = x * 2
    pdb.set_trace()  # Debugger stops here
    result += 10
    return result

# Commands in pdb:
# n - next line
# s - step into function
# c - continue
# p variable - print variable
# l - list source code
# q - quit

# Conditional breakpoint
def process_items(items):
    for i, item in enumerate(items):
        if i == 5:
            pdb.set_trace()  # Stop at 5th iteration
        process(item)

# Post-mortem debugging
def buggy_function():
    raise ValueError("Something went wrong")

try:
    buggy_function()
except:
    import pdb
    pdb.post_mortem()  # Debug at exception point

# logging for debugging
import logging

logging.basicConfig(level=logging.DEBUG)

def calculate(x, y):
    logging.debug(f"Calculating {x} + {y}")
    result = x + y
    logging.info(f"Result: {result}")
    return result

# Rich tracebacks
from rich.traceback import install
install(show_locals=True)

# Debug decorator
def debug(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        print(f"Args: {args}, Kwargs: {kwargs}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper

@debug
def add(a, b):
    return a + b

---

### Q99: How do you use logging effectively in Python?

**Answer:**
Configure proper log levels, use formatters, and implement log rotation.

```python
import logging
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# Basic logging configuration
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Log levels
logging.debug("Debug message")
logging.info("Info message")
logging.warning("Warning message")
logging.error("Error message")
logging.critical("Critical message")

# Logger per module
logger = logging.getLogger(__name__)

def my_function():
    logger.info("Function called")
    try:
        risky_operation()
    except Exception as e:
        logger.exception("Operation failed")  # Logs with traceback

# File logging with rotation
file_handler = RotatingFileHandler(
    'app.log',
    maxBytes=10*1024*1024,  # 10MB
    backupCount=5
)
file_handler.setLevel(logging.INFO)

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Formatter
formatter = logging.Formatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
file_handler.setFormatter(formatter)
console_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Contextual logging
logger.info("User %s logged in", username)
logger.error("Failed to process order %s", order_id, exc_info=True)

# Structured logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            'timestamp': self.formatTime(record),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module
        }
        return json.dumps(log_data)

# Time-based rotation
timed_handler = TimedRotatingFileHandler(
    'app.log',
    when='midnight',
    interval=1,
    backupCount=7
)

---

### Q100: How do you implement code quality checks?

**Answer:**
Use linters, formatters, type checkers, and automated quality tools.

```python
# Black - code formatter
# Install: pip install black
# Run: black myfile.py

# Flake8 - style guide enforcement
# Install: pip install flake8
# Run: flake8 myfile.py

# pylint - comprehensive linting
# Install: pip install pylint
# Run: pylint myfile.py

# mypy - type checking
# Install: pip install mypy
# Run: mypy myfile.py

# Example with type hints
from typing import List, Dict, Optional

def process_data(
    items: List[int],
    config: Dict[str, str],
    threshold: Optional[int] = None
) -> List[int]:
    """Process items based on configuration.
    
    Args:
        items: List of integers to process
        config: Configuration dictionary
        threshold: Optional filtering threshold
    
    Returns:
        Processed list of integers
    """
    result = []
    for item in items:
        if threshold is None or item > threshold:
            result.append(item * 2)
    return result

# pre-commit hooks for automation
# .pre-commit-config.yaml:
# repos:
#   - repo: https://github.com/psf/black
#     hooks:
#       - id: black
#   - repo: https://github.com/PyCQA/flake8
#     hooks:
#       - id: flake8

# isort - import sorting
# Install: pip install isort
# Run: isort myfile.py

# bandit - security linter
# Install: pip install bandit
# Run: bandit -r myproject/

---

### Q101: How do you handle exceptions and errors properly?

**Answer:**
Use specific exceptions, proper exception hierarchies, and appropriate error handling.

```python
# Specific exceptions
# Bad
try:
    risky_operation()
except Exception:  # Too broad!
    pass

# Good
try:
    risky_operation()
except (ValueError, TypeError) as e:
    logger.error(f"Invalid input: {e}")
except IOError as e:
    logger.error(f"I/O error: {e}")

# Custom exception hierarchy
class ApplicationError(Exception):
    """Base exception for application"""
    pass

class ValidationError(ApplicationError):
    """Data validation failed"""
    pass

class DatabaseError(ApplicationError):
    """Database operation failed"""
    pass

class ConnectionError(DatabaseError):
    """Database connection failed"""
    pass

# Using custom exceptions
def validate_user(user_data):
    if not user_data.get('email'):
        raise ValidationError("Email is required")
    if not user_data.get('age') or user_data['age'] < 0:
        raise ValidationError("Valid age is required")

# Exception chaining
try:
    connect_to_database()
except ConnectionError as e:
    raise DatabaseError("Failed to initialize database") from e

# Context managers for cleanup
class ManagedResource:
    def __enter__(self):
        self.resource = acquire_resource()
        return self.resource
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        release_resource(self.resource)
        return False  # Don't suppress exceptions

# Proper exception handling pattern
def process_file(filename):
    try:
        with open(filename) as f:
            data = f.read()
            return process_data(data)
    except FileNotFoundError:
        logger.error(f"File not found: {filename}")
        return None
    except PermissionError:
        logger.error(f"Permission denied: {filename}")
        return None
    except Exception as e:
        logger.exception(f"Unexpected error processing {filename}")
        raise  # Re-raise unexpected exceptions

# Don't catch exceptions you can't handle
def bad_example():
    try:
        critical_operation()
    except Exception:
        pass  # BAD - silently swallows errors!

---

### Q102: How do you write doctests in Python?

**Answer:**
Embed tests in docstrings for executable documentation and simple tests.

```python
def add(a, b):
    """Add two numbers.
    
    >>> add(2, 3)
    5
    >>> add(-1, 1)
    0
    >>> add(0, 0)
    0
    """
    return a + b

# Run doctests
# python -m doctest mymodule.py -v

# More complex examples
def factorial(n):
    """Calculate factorial of n.
    
    >>> factorial(5)
    120
    >>> factorial(0)
    1
    >>> factorial(-1)
    Traceback (most recent call last):
        ...
    ValueError: Factorial not defined for negative numbers
    """
    if n < 0:
        raise ValueError("Factorial not defined for negative numbers")
    if n == 0:
        return 1
    return n * factorial(n - 1)

# Testing with setup
def process_list(items):
    """Process list of items.
    
    >>> items = [1, 2, 3, 4, 5]
    >>> process_list(items)
    [2, 4, 6, 8, 10]
    >>> process_list([])
    []
    """
    return [x * 2 for x in items]

# In test file
if __name__ == "__main__":
    import doctest
    doctest.testmod(verbose=True)

---

### Q103: How do you implement integration tests?

**Answer:**
Test multiple components together, use test databases, and mock external services.

```python
import pytest
from unittest.mock import patch, Mock

# Database integration test
class TestUserRepository:
    @pytest.fixture
    def db_connection(self):
        """Setup test database"""
        conn = create_test_db()
        setup_schema(conn)
        yield conn
        teardown_db(conn)
    
    def test_create_user(self, db_connection):
        repo = UserRepository(db_connection)
        user = repo.create_user("alice@example.com", "Alice")
        
        # Verify in database
        saved_user = repo.find_by_email("alice@example.com")
        assert saved_user.name == "Alice"
    
    def test_update_user(self, db_connection):
        repo = UserRepository(db_connection)
        user = repo.create_user("bob@example.com", "Bob")
        
        repo.update_user(user.id, name="Robert")
        updated = repo.find_by_id(user.id)
        assert updated.name == "Robert"

# API integration test
@pytest.fixture
def api_client():
    from flask import Flask
    app = create_app(config='testing')
    return app.test_client()

def test_api_endpoint(api_client):
    response = api_client.post('/api/users', json={
        'email': 'test@example.com',
        'name': 'Test User'
    })
    assert response.status_code == 201
    assert response.json['email'] == 'test@example.com'

# Mock external services
def test_with_external_api():
    with patch('requests.get') as mock_get:
        mock_get.return_value.json.return_value = {'data': 'test'}
        mock_get.return_value.status_code = 200
        
        result = fetch_external_data()
        assert result == {'data': 'test'}

---

### Q104: How do you test asynchronous code?

**Answer:**
Use pytest-asyncio or unittest async support for testing coroutines.

```python
import pytest
import asyncio

# pytest-asyncio
@pytest.mark.asyncio
async def test_async_function():
    result = await async_operation()
    assert result == expected_value

# Test with timeout
@pytest.mark.asyncio
@pytest.mark.timeout(5)
async def test_with_timeout():
    result = await slow_async_operation()
    assert result is not None

# Mock async functions
@pytest.mark.asyncio
async def test_async_api():
    with patch('aiohttp.ClientSession.get') as mock_get:
        mock_response = Mock()
        mock_response.json = asyncio.coroutine(lambda: {'data': 'test'})
        mock_get.return_value.__aenter__.return_value = mock_response
        
        result = await fetch_data('https://api.example.com')
        assert result == {'data': 'test'}

# unittest async support (Python 3.8+)
import unittest

class TestAsync(unittest.IsolatedAsyncioTestCase):
    async def test_async_operation(self):
        result = await async_function()
        self.assertEqual(result, expected)
    
    async def asyncSetUp(self):
        """Async setup"""
        self.resource = await create_async_resource()
    
    async def asyncTearDown(self):
        """Async teardown"""
        await self.resource.cleanup()

# Test concurrent operations
@pytest.mark.asyncio
async def test_concurrent_requests():
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    assert len(results) == len(urls)

---

### Q105: How do you implement property-based testing?

**Answer:**
Use Hypothesis to generate test cases automatically and find edge cases.

```python
from hypothesis import given, strategies as st
import pytest

# Basic property test
@given(st.integers(), st.integers())
def test_addition_commutative(a, b):
    assert a + b == b + a

# Test with constraints
@given(st.integers(min_value=0, max_value=100))
def test_square_positive(n):
    assert n ** 2 >= 0

# Test with lists
@given(st.lists(st.integers()))
def test_reverse_twice(lst):
    assert list(reversed(list(reversed(lst)))) == lst

# Test with strings
@given(st.text())
def test_string_upper_lower(s):
    assert s.upper().lower() == s.lower()

# Complex data structures
@given(st.dictionaries(
    keys=st.text(min_size=1),
    values=st.integers()
))
def test_dict_operations(d):
    keys = list(d.keys())
    assert all(k in d for k in keys)

# Custom strategies
email_strategy = st.emails()
url_strategy = st.from_regex(r'https://[a-z]+\.com')

@given(email_strategy)
def test_email_validation(email):
    assert '@' in email

# Example finding bugs
@given(st.lists(st.integers()))
def test_sort_invariant(lst):
    sorted_lst = sorted(lst)
    # Properties of sorted list
    assert len(sorted_lst) == len(lst)
    assert set(sorted_lst) == set(lst)
    # Sorted property
    for i in range(len(sorted_lst) - 1):
        assert sorted_lst[i] <= sorted_lst[i + 1]

---

### Q106: How do you test database operations?

**Answer:**
Use test databases, transactions, and fixtures for isolated database tests.

```python
import pytest
import sqlite3

# Fixture for test database
@pytest.fixture
def db():
    """Create test database"""
    conn = sqlite3.connect(':memory:')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE users (
            id INTEGER PRIMARY KEY,
            email TEXT UNIQUE,
            name TEXT
        )
    ''')
    conn.commit()
    yield conn
    conn.close()

# Test database operations
def test_insert_user(db):
    cursor = db.cursor()
    cursor.execute(
        "INSERT INTO users (email, name) VALUES (?, ?)",
        ("alice@example.com", "Alice")
    )
    db.commit()
    
    cursor.execute("SELECT * FROM users WHERE email = ?", ("alice@example.com",))
    user = cursor.fetchone()
    assert user[2] == "Alice"

# Transaction rollback in tests
@pytest.fixture
def db_transaction(db):
    """Each test runs in transaction"""
    cursor = db.cursor()
    cursor.execute("BEGIN")
    yield db
    db.rollback()

def test_with_rollback(db_transaction):
    # Changes are rolled back after test
    cursor = db_transaction.cursor()
    cursor.execute("INSERT INTO users (email, name) VALUES (?, ?)",
                  ("test@example.com", "Test"))

# Test with real database (PostgreSQL example)
@pytest.fixture(scope='session')
def postgres_db():
    # Create test database
    conn = psycopg2.connect("dbname=test_db")
    setup_schema(conn)
    yield conn
    teardown_db(conn)
    conn.close()

# Test repository pattern
class TestUserRepository:
    def test_find_by_email(self, db):
        repo = UserRepository(db)
        user = repo.create("alice@example.com", "Alice")
        
        found = repo.find_by_email("alice@example.com")
        assert found.name == "Alice"

---

### Q107: How do you implement code coverage analysis?

**Answer:**
Use coverage.py to measure test coverage and identify untested code.

```python
# Install: pip install coverage pytest-cov

# Run with coverage
# pytest --cov=mymodule tests/
# coverage run -m pytest
# coverage report
# coverage html  # Generate HTML report

# Configuration in .coveragerc or pyproject.toml
# [coverage:run]
# source = mymodule
# omit = */tests/*

# Example with coverage decorator
import coverage

def with_coverage(func):
    def wrapper(*args, **kwargs):
        cov = coverage.Coverage()
        cov.start()
        try:
            result = func(*args, **kwargs)
        finally:
            cov.stop()
            cov.save()
            cov.report()
        return result
    return wrapper

# Aim for meaningful coverage, not 100%
# Focus on:
# - Critical business logic
# - Error handling paths
# - Edge cases

# Example test showing coverage
def calculate_discount(price, customer_type):
    """
    Calculate discount based on customer type.
    This function has multiple branches to test.
    """
    if price < 0:
        raise ValueError("Price cannot be negative")
    
    if customer_type == "premium":
        return price * 0.8  # 20% discount
    elif customer_type == "regular":
        return price * 0.95  # 5% discount
    else:
        return price  # No discount

# Tests for full coverage
def test_discount_premium():
    assert calculate_discount(100, "premium") == 80

def test_discount_regular():
    assert calculate_discount(100, "regular") == 95

def test_discount_none():
    assert calculate_discount(100, "guest") == 100

def test_negative_price():
    with pytest.raises(ValueError):
        calculate_discount(-10, "premium")

---

### Q108: How do you debug performance issues?

**Answer:**
Use profilers, timing analysis, and memory profilers to identify bottlenecks.

```python
import cProfile
import pstats
import time
from memory_profiler import profile

# CPU profiling
def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Code to profile
    slow_function()
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(10)

# Line-by-line profiling
# pip install line_profiler
# @profile
def detailed_profiling():
    result = []
    for i in range(1000):
        result.append(i ** 2)
    return result

# Memory profiling
# @profile
def memory_intensive():
    big_list = [i for i in range(1000000)]
    return sum(big_list)

# Custom timing context manager
from contextlib import contextmanager
import time

@contextmanager
def timing(label):
    start = time.perf_counter()
    try:
        yield
    finally:
        end = time.perf_counter()
        print(f"{label}: {end - start:.4f}s")

# Usage
with timing("Database query"):
    results = db.query("SELECT * FROM large_table")

# Identify slow operations
import logging

def logged_timing(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        if elapsed > 1.0:  # Log if > 1 second
            logging.warning(f"{func.__name__} took {elapsed:.2f}s")
        return result
    return wrapper

# Find memory leaks
import gc
import sys

def find_memory_leaks():
    # Force garbage collection
    gc.collect()
    
    # Get all objects
    objects = gc.get_objects()
    
    # Find specific types
    leaked_lists = [obj for obj in objects if isinstance(obj, list)]
    print(f"Found {len(leaked_lists)} list objects")

---

### Q109: How do you implement continuous integration for Python?

**Answer:**
Use CI/CD tools with automated testing, linting, and quality checks.

```python
# .github/workflows/python-app.yml (GitHub Actions)
"""
name: Python application

on: [push, pull_request]

jobs:
  build:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v2
    - name: Set up Python
      uses: actions/setup-python@v2
      with:
        python-version: '3.11'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
        pip install pytest pytest-cov flake8 black mypy
    
    - name: Lint with flake8
      run: |
        flake8 . --count --select=E9,F63,F7,F82 --show-source --statistics
    
    - name: Check formatting with black
      run: black --check .
    
    - name: Type check with mypy
      run: mypy src/
    
    - name: Test with pytest
      run: |
        pytest --cov=src tests/
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
"""

# tox.ini for testing multiple Python versions
"""
[tox]
envlist = py39,py310,py311

[testenv]
deps =
    pytest
    pytest-cov
commands =
    pytest tests/ --cov=src
"""

# pre-commit configuration
"""
# .pre-commit-config.yaml
repos:
  - repo: https://github.com/psf/black
    rev: 23.1.0
    hooks:
      - id: black
  
  - repo: https://github.com/PyCQA/flake8
    rev: 6.0.0
    hooks:
      - id: flake8
  
  - repo: https://github.com/pre-commit/mirrors-mypy
    rev: v1.0.0
    hooks:
      - id: mypy
"""

---

### Q110: How do you document Python code effectively?

**Answer:**
Use docstrings, type hints, comments, and documentation generators.

```python
from typing import List, Optional, Dict

def process_data(
    items: List[int],
    threshold: Optional[int] = None,
    config: Dict[str, str] = None
) -> List[int]:
    """Process list of items based on threshold and configuration.
    
    This function filters and transforms items according to the provided
    threshold value and configuration parameters.
    
    Args:
        items: List of integers to process
        threshold: Optional filtering threshold. Items below this value
                  are excluded. Defaults to None (no filtering).
        config: Optional configuration dictionary. Defaults to None.
    
    Returns:
        Processed list of integers, filtered and transformed.
    
    Raises:
        ValueError: If items list is empty
        TypeError: If items contains non-integer values
    
    Examples:
        >>> process_data([1, 2, 3, 4, 5], threshold=3)
        [4, 5]
        
        >>> process_data([10, 20, 30])
        [10, 20, 30]
    
    Note:
        This function modifies data in-place for efficiency.
    
    See Also:
        validate_data: For pre-processing validation
    """
    if not items:
        raise ValueError("Items list cannot be empty")
    
    # Apply threshold filter if provided
    if threshold is not None:
        items = [x for x in items if x >= threshold]
    
    # Apply configuration transforms
    if config and config.get('double'):
        items = [x * 2 for x in items]
    
    return items

# Class documentation
class DataProcessor:
    """Process and transform data streams.
    
    This class provides methods for processing data streams with
    various transformation and filtering options.
    
    Attributes:
        buffer_size: Size of internal buffer (default: 1024)
        encoding: Character encoding for text data (default: 'utf-8')
    
    Example:
        >>> processor = DataProcessor(buffer_size=2048)
        >>> result = processor.process([1, 2, 3])
    """
    
    def __init__(self, buffer_size: int = 1024):
        """Initialize DataProcessor.
        
        Args:
            buffer_size: Size of internal buffer in bytes
        """
        self.buffer_size = buffer_size
    
    def process(self, data: List[int]) -> List[int]:
        """Process data using configured settings."""
        pass

# Generate documentation with Sphinx
# Install: pip install sphinx
# sphinx-quickstart
# make html

---

## Section 8: Modern Python Features & Best Practices (Q111-125)

### Q111: What are Python 3.10+ new features?

**Answer:**
Structural pattern matching, better error messages, union types, and more.

```python
# Structural pattern matching (Python 3.10+)
def process_command(command):
    match command:
        case ["quit"]:
            return "Exiting..."
        case ["load", filename]:
            return f"Loading {filename}"
        case ["save", filename]:
            return f"Saving to {filename}"
        case ["delete", *files]:
            return f"Deleting {len(files)} files"
        case _:
            return "Unknown command"

# Pattern matching with classes
from dataclasses import dataclass

@dataclass
class Point:
    x: int
    y: int

def location(point):
    match point:
        case Point(x=0, y=0):
            return "Origin"
        case Point(x=0, y=y):
            return f"On Y-axis at {y}"
        case Point(x=x, y=0):
            return f"On X-axis at {x}"
        case Point(x=x, y=y):
            return f"At ({x}, {y})"

# Union types with | operator (Python 3.10+)
def process(value: int | str) -> int | None:
    if isinstance(value, int):
        return value * 2
    return None

# Better error messages (Python 3.10+)
# dictionary = {"key": "value"}
# print(dictionary["wrong_key"])
# KeyError: 'wrong_key'. Did you mean: 'key'?

# Parenthesized context managers (Python 3.10+)
with (
    open('file1.txt') as f1,
    open('file2.txt') as f2
):
    process(f1, f2)

---

### Q112: What are Python 3.11+ performance improvements?

**Answer:**
Faster interpreter, better error messages, exception groups, and task groups.

```python
# Python 3.11 is 10-60% faster than 3.10!

# Exception groups (Python 3.11+)
# try:
#     raise ExceptionGroup("Multiple errors", [
#         ValueError("Invalid value"),
#         TypeError("Wrong type")
#     ])
# except* ValueError as e:
#     print(f"Handling ValueError: {e}")
# except* TypeError as e:
#     print(f"Handling TypeError: {e}")

# Task groups for asyncio (Python 3.11+)
import asyncio

async def main():
    async with asyncio.TaskGroup() as tg:
        task1 = tg.create_task(fetch_data("url1"))
        task2 = tg.create_task(fetch_data("url2"))
    # Both tasks complete or all cancelled

# Better error locations
# def calculate():
#     result = some_function(param1, param2, param3)
# Python 3.11 shows exactly which parameter caused the error!

# TOML support in stdlib (Python 3.11+)
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

---

### Q113: How do you use dataclasses effectively?

**Answer:**
Dataclasses reduce boilerplate for data-focused classes.

```python
from dataclasses import dataclass, field, asdict, astuple
from typing import List
from datetime import datetime

# Basic dataclass
@dataclass
class Person:
    name: str
    age: int
    email: str

person = Person("Alice", 30, "alice@example.com")
print(person)  # Person(name='Alice', age=30, email='alice@example.com')

# With default values
@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0
    tags: List[str] = field(default_factory=list)

# Immutable dataclass
@dataclass(frozen=True)
class Point:
    x: int
    y: int

# Can be used as dict key
points = {Point(0, 0): "origin"}

# Ordered comparison
@dataclass(order=True)
class Student:
    name: str = field(compare=False)
    grade: float

students = [Student("Bob", 85), Student("Alice", 92)]
print(sorted(students))  # Sorted by grade

# Post-initialization
@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)
    
    def __post_init__(self):
        self.area = self.width * self.height

# Convert to dict/tuple
person = Person("Bob", 25, "bob@example.com")
print(asdict(person))  # {'name': 'Bob', 'age': 25, 'email': '...'}
print(astuple(person))  # ('Bob', 25, 'bob@example.com')

# Inheritance
@dataclass
class Employee(Person):
    employee_id: str
    department: str

---

### Q114: What are type hints best practices?

**Answer:**
Use gradual typing, proper annotations, and type checkers for better code quality.

```python
from typing import (
    List, Dict, Tuple, Set, Optional, Union,
    Callable, Any, TypeVar, Generic, Protocol
)

# Basic types
def greet(name: str) -> str:
    return f"Hello, {name}"

# Collections
def process_items(items: List[int]) -> Dict[str, int]:
    return {"count": len(items), "sum": sum(items)}

# Optional (can be None)
def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)

# Union types
def process(value: Union[int, str, List[int]]) -> str:
    return str(value)

# Modern union syntax (Python 3.10+)
def process_modern(value: int | str | list[int]) -> str:
    return str(value)

# Callable types
def apply(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

# Generic types
T = TypeVar('T')

def first(items: List[T]) -> Optional[T]:
    return items[0] if items else None

# Generic class
class Stack(Generic[T]):
    def __init__(self) -> None:
        self._items: List[T] = []
    
    def push(self, item: T) -> None:
        self._items.append(item)
    
    def pop(self) -> T:
        return self._items.pop()

# Protocol for structural typing
class Drawable(Protocol):
    def draw(self) -> str:
        ...

def render(obj: Drawable) -> None:
    print(obj.draw())

# TypedDict for structured dicts
from typing import TypedDict

class UserDict(TypedDict):
    name: str
    age: int
    email: str

def create_user(data: UserDict) -> None:
    print(data['name'])

# Type aliases
UserId = int
UserName = str
UserMap = Dict[UserId, UserName]

users: UserMap = {1: "Alice", 2: "Bob"}

---

### Q115: How do you use context managers effectively?

**Answer:**
Create custom context managers for resource management and cleanup.

```python
from contextlib import contextmanager, ExitStack
import time

# Function-based context manager
@contextmanager
def timer(label: str):
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{label}: {end - start:.4f}s")

with timer("Operation"):
    time.sleep(1)

# Class-based context manager
class DatabaseConnection:
    def __init__(self, connection_string: str):
        self.connection_string = connection_string
    
    def __enter__(self):
        print(f"Connecting to {self.connection_string}")
        self.connection = f"Connection({self.connection_string})"
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection")
        return False  # Propagate exceptions

# Multiple context managers
with DatabaseConnection("db1") as db1, \
     DatabaseConnection("db2") as db2:
    process(db1, db2)

# ExitStack for dynamic context managers
def process_files(filenames):
    with ExitStack() as stack:
        files = [stack.enter_context(open(fn)) for fn in filenames]
        for f in files:
            process(f)
    # All files automatically closed

# Suppress exceptions
from contextlib import suppress

with suppress(FileNotFoundError):
    os.remove("nonexistent_file.txt")

# Temporary directory
@contextmanager
def temporary_directory():
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

---

### Q116: What are the best practices for Python project structure?

**Answer:**
Follow standard layouts, use proper packaging, and organize code logically.

```python
"""
Recommended project structure:

myproject/
├── src/
│   └── mypackage/
│       ├── __init__.py
│       ├── core.py
│       ├── utils.py
│       └── models/
│           ├── __init__.py
│           └── user.py
├── tests/
│   ├── __init__.py
│   ├── test_core.py
│   └── test_utils.py
├── docs/
│   ├── conf.py
│   └── index.rst
├── pyproject.toml
├── setup.py
├── README.md
├── LICENSE
├── .gitignore
└── requirements.txt
"""

# pyproject.toml (modern Python packaging)
"""
[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "mypackage"
version = "0.1.0"
description = "My Python package"
authors = [{name = "Your Name", email = "you@example.com"}]
dependencies = [
    "requests>=2.28.0",
    "numpy>=1.24.0"
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "black>=23.0.0",
    "mypy>=1.0.0"
]
"""

# __init__.py
"""
from .core import main_function
from .utils import helper_function

__version__ = "0.1.0"
__all__ = ["main_function", "helper_function"]
"""

---

### Q117: How do you handle configuration in Python applications?

**Answer:**
Use configuration files, environment variables, and proper config management.

```python
# Using environment variables
import os
from pathlib import Path

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///default.db")
DEBUG = os.getenv("DEBUG", "False").lower() == "true"

# Using .env files with python-dotenv
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

API_KEY = os.getenv("API_KEY")
SECRET_KEY = os.getenv("SECRET_KEY")

# Configuration class
from dataclasses import dataclass

@dataclass
class Config:
    DATABASE_URL: str
    DEBUG: bool
    API_KEY: str
    MAX_CONNECTIONS: int = 10
    
    @classmethod
    def from_env(cls):
        return cls(
            DATABASE_URL=os.getenv("DATABASE_URL"),
            DEBUG=os.getenv("DEBUG", "False").lower() == "true",
            API_KEY=os.getenv("API_KEY"),
            MAX_CONNECTIONS=int(os.getenv("MAX_CONNECTIONS", "10"))
        )

config = Config.from_env()

# YAML configuration
import yaml

with open("config.yaml") as f:
    config = yaml.safe_load(f)

# TOML configuration (Python 3.11+)
import tomllib

with open("config.toml", "rb") as f:
    config = tomllib.load(f)

# Pydantic for validation
from pydantic import BaseSettings

class Settings(BaseSettings):
    database_url: str
    api_key: str
    debug: bool = False
    
    class Config:
        env_file = ".env"

settings = Settings()

# Configuration with different environments
class Config:
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False

class TestingConfig(Config):
    TESTING = True

config = {
    "development": DevelopmentConfig,
    "production": ProductionConfig,
    "testing": TestingConfig
}

app_config = config[os.getenv("ENV", "development")]

---

### Q118: How do you implement command-line interfaces (CLI)?

**Answer:**
Use argparse, click, or typer for creating robust CLI applications.

```python
# argparse - standard library
import argparse

def main():
    parser = argparse.ArgumentParser(description="Process some data")
    
    parser.add_argument("input", help="Input file")
    parser.add_argument("-o", "--output", help="Output file")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("--format", choices=["json", "csv"], default="json")
    
    args = parser.parse_args()
    
    if args.verbose:
        print(f"Processing {args.input}")

if __name__ == "__main__":
    main()

# click - popular third-party
import click

@click.command()
@click.argument("input")
@click.option("--output", "-o", help="Output file")
@click.option("--verbose", "-v", is_flag=True)
@click.option("--format", type=click.Choice(["json", "csv"]))
def cli(input, output, verbose, format):
    """Process some data."""
    if verbose:
        click.echo(f"Processing {input}")

if __name__ == "__main__":
    cli()

# typer - modern with type hints
import typer
from pathlib import Path# 150 Advanced Python Questions & Answers - Complete 2026 Mastery Guide

## Table of Contents
1. Core Fundamentals & Data Structures (Q1-20)
2. Object-Oriented Programming & Design Patterns (Q21-35)
3. Functional Programming & Advanced Functions (Q36-50)
4. Iterators, Generators & Memory Optimization (Q51-65)
5. Decorators, Metaclasses & Descriptors (Q66-80)
6. Concurrency, Parallelism & Async Programming (Q81-95)
7. Performance Optimization & Profiling (Q96-110)
8. Testing, Debugging & Code Quality (Q111-120)
9. Modern Python Features & Best Practices (Q121-135)
10. Real-World Applications & Industry Standards (Q136-150)

---

## Section 1: Core Fundamentals & Data Structures (Q1-20)

### Q1: What are the key differences between Python's mutable and immutable types, and why does it matter?

**Answer:**
Mutable types can be changed after creation (lists, dicts, sets), while immutable types cannot (strings, tuples, integers, frozensets).

```python
# Mutable - list can be modified
my_list = [1, 2, 3]
my_list[0] = 100  # Works fine
print(my_list)  # [100, 2, 3]

# Immutable - tuple cannot be modified
my_tuple = (1, 2, 3)
# my_tuple[0] = 100  # TypeError!

# Why it matters: dictionary keys must be immutable
valid_dict = {(1, 2): "tuple key"}  # Valid
# invalid_dict = {[1, 2]: "list key"}  # TypeError!

# Memory implications
x = [1, 2, 3]
y = x  # Both point to same object
y.append(4)
print(x)  # [1, 2, 3, 4] - x is also modified!

# Immutable creates new objects
a = "hello"
b = a
b = b + " world"  # Creates new string
print(a)  # "hello" - unchanged
```

**Why it matters:** Affects function arguments, dictionary keys, performance, and prevents unexpected side effects.

---

### Q2: How does Python's memory management and garbage collection work?

**Answer:**
Python uses reference counting combined with a cycle-detecting garbage collector.

```python
import sys
import gc

# Reference counting
x = [1, 2, 3]
print(sys.getrefcount(x))  # Shows reference count

y = x  # Increases ref count
print(sys.getrefcount(x))

del y  # Decreases ref count
print(sys.getrefcount(x))

# Circular references - where cycle detector helps
class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

node1 = Node(1)
node2 = Node(2)
node1.next = node2
node2.next = node1  # Circular reference!

# Manual garbage collection
gc.collect()  # Forces collection of circular references

# Check garbage collection stats
print(gc.get_count())  # Shows collection counts per generation

# Disable/enable garbage collection (rare cases)
gc.disable()
# ... performance-critical code ...
gc.enable()
```

**Key points:** Objects are deleted when ref count reaches 0. Generational GC handles circular references efficiently.

---

### Q3: What is the difference between `is` and `==`, and when should you use each?

**Answer:**
`is` checks object identity (same memory location), `==` checks value equality.

```python
# Integer caching (Python caches small integers)
a = 256
b = 256
print(a is b)  # True - same cached object

a = 257
b = 257
print(a is b)  # False in some contexts - different objects

# String interning
s1 = "hello"
s2 = "hello"
print(s1 is s2)  # True - strings are interned

# Lists - never use 'is' for value comparison
list1 = [1, 2, 3]
list2 = [1, 2, 3]
print(list1 == list2)  # True - same values
print(list1 is list2)  # False - different objects

# Best practice: use 'is' only for None, True, False
value = None
if value is None:  # Correct
    print("Value is None")

if value == None:  # Works but not idiomatic
    print("Value is None")

# Identity check for singleton pattern
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True - same instance
```

---

### Q4: How do Python's data structures (list, tuple, set, dict) differ in performance and use cases?

**Answer:**
Each structure is optimized for different operations:

```python
import timeit

# LIST - ordered, mutable, allows duplicates
# Best for: ordered collections, frequent indexing
my_list = [1, 2, 3, 4, 5]
my_list.append(6)  # O(1) average
my_list.insert(0, 0)  # O(n) - shifts all elements
print(3 in my_list)  # O(n) - linear search

# TUPLE - ordered, immutable, allows duplicates
# Best for: fixed collections, dictionary keys, memory efficiency
my_tuple = (1, 2, 3, 4, 5)
# my_tuple.append(6)  # Error - immutable
print(3 in my_tuple)  # O(n) - linear search
# 20-30% less memory than lists

# SET - unordered, mutable, no duplicates
# Best for: membership testing, removing duplicates, set operations
my_set = {1, 2, 3, 4, 5}
my_set.add(6)  # O(1) average
print(3 in my_set)  # O(1) - hash lookup!

# Remove duplicates efficiently
duplicates = [1, 2, 2, 3, 3, 3, 4]
unique = list(set(duplicates))

# DICT - key-value pairs, unordered (ordered in Python 3.7+)
# Best for: fast lookups, counting, caching
my_dict = {"a": 1, "b": 2, "c": 3}
print(my_dict["a"])  # O(1) - hash lookup
my_dict["d"] = 4  # O(1) average

# Performance comparison
def test_list():
    return 5000 in list(range(10000))

def test_set():
    return 5000 in set(range(10000))

print("List lookup:", timeit.timeit(test_list, number=10000))
print("Set lookup:", timeit.timeit(test_set, number=10000))
# Set is dramatically faster for membership testing!

# Use case examples
# List: maintaining order of items
tasks = ["task1", "task2", "task3"]

# Tuple: function returns multiple values
def get_coordinates():
    return (10, 20)

# Set: tracking unique visitors
visitors = set()
visitors.add("user1")
visitors.add("user1")  # Duplicate ignored
print(len(visitors))  # 1

# Dict: configuration, caching
config = {
    "host": "localhost",
    "port": 8080,
    "debug": True
}
```

---

### Q5: What are dictionary comprehensions and when should you use them over loops?

**Answer:**
Dictionary comprehensions provide concise syntax for creating dictionaries and are generally faster than loops.

```python
# Basic dictionary comprehension
squares = {x: x**2 for x in range(10)}
print(squares)  # {0: 0, 1: 1, 2: 4, ...}

# With conditional
even_squares = {x: x**2 for x in range(10) if x % 2 == 0}

# From two lists
keys = ['a', 'b', 'c']
values = [1, 2, 3]
my_dict = {k: v for k, v in zip(keys, values)}

# Transforming existing dictionary
prices = {"apple": 0.50, "banana": 0.30, "orange": 0.70}
increased_prices = {item: price * 1.1 for item, price in prices.items()}

# Filtering dictionary
filtered = {k: v for k, v in prices.items() if v > 0.40}

# Swapping keys and values
inverted = {v: k for k, v in my_dict.items()}

# Real-world example: counting occurrences
text = "hello world hello python world"
word_count = {word: text.split().count(word) for word in set(text.split())}

# Better way using Counter (more efficient)
from collections import Counter
word_count = Counter(text.split())

# Nested dictionary comprehension
matrix = {
    i: {j: i * j for j in range(3)}
    for i in range(3)
}
# {0: {0: 0, 1: 0, 2: 0}, 1: {0: 0, 1: 1, 2: 2}, 2: {0: 0, 1: 2, 2: 4}}

# When NOT to use comprehensions
# Too complex - hurts readability
# bad = {k: complex_function(k) if condition1(k) else other_function(k) 
#        for k in items if condition2(k) and condition3(k)}

# Better as traditional loop
result = {}
for k in items:
    if condition2(k) and condition3(k):
        if condition1(k):
            result[k] = complex_function(k)
        else:
            result[k] = other_function(k)
```

---

### Q6: How do you properly handle mutable default arguments in Python?

**Answer:**
Mutable defaults are created once at function definition, causing unexpected behavior. Use None as default instead.

```python
# WRONG - mutable default argument bug
def add_item_wrong(item, items=[]):
    items.append(item)
    return items

print(add_item_wrong(1))  # [1]
print(add_item_wrong(2))  # [1, 2] - Unexpected!
print(add_item_wrong(3))  # [1, 2, 3] - Same list!

# CORRECT - use None as default
def add_item_correct(item, items=None):
    if items is None:
        items = []
    items.append(item)
    return items

print(add_item_correct(1))  # [1]
print(add_item_correct(2))  # [2] - Fresh list each time
print(add_item_correct(3))  # [3]

# Real-world example: database connection pool
class Database:
    def __init__(self, connections=None):
        if connections is None:
            connections = []
        self.connections = connections
    
    def add_connection(self, conn):
        self.connections.append(conn)

# Why it happens
def show_default(x=[]):
    print(id(x))  # Same ID every call - same object!
    x.append(1)
    return x

show_default()
show_default()
show_default()  # All use same list object

# Alternative: using factory function
from typing import List

def process_items(items: List[int] = None) -> List[int]:
    items = items or []  # Common idiom
    return [x * 2 for x in items]

# Or for more complex defaults
from typing import Dict

def create_config(options: Dict[str, any] = None) -> Dict[str, any]:
    if options is None:
        options = {
            "timeout": 30,
            "retries": 3,
            "debug": False
        }
    return options
```

---

### Q7: What are Python's special methods (dunder methods) and how do you use them?

**Answer:**
Special methods let you define behavior for built-in operations on custom objects.

```python
class Vector:
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # String representation
    def __repr__(self):
        return f"Vector({self.x}, {self.y})"
    
    def __str__(self):
        return f"({self.x}, {self.y})"
    
    # Arithmetic operations
    def __add__(self, other):
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        return Vector(self.x * scalar, self.y * scalar)
    
    # Comparison operations
    def __eq__(self, other):
        return self.x == other.x and self.y == other.y
    
    def __lt__(self, other):
        return (self.x**2 + self.y**2) < (other.x**2 + other.y**2)
    
    # Container operations
    def __len__(self):
        return 2
    
    def __getitem__(self, index):
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Index out of range")
    
    # Context manager
    def __enter__(self):
        print("Entering context")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Exiting context")
        return False
    
    # Callable
    def __call__(self, scalar):
        return self * scalar

# Usage
v1 = Vector(1, 2)
v2 = Vector(3, 4)

print(v1)  # Calls __str__: (1, 2)
print(repr(v1))  # Calls __repr__: Vector(1, 2)

v3 = v1 + v2  # Calls __add__
print(v3)  # (4, 6)

print(v1 == Vector(1, 2))  # Calls __eq__: True
print(v1 < v2)  # Calls __lt__: True

print(len(v1))  # Calls __len__: 2
print(v1[0])  # Calls __getitem__: 1

with v1:  # Calls __enter__ and __exit__
    print("Inside context")

result = v1(5)  # Calls __call__
print(result)  # (5, 10)

# Practical example: custom dictionary-like class
class CaseInsensitiveDict:
    def __init__(self):
        self._data = {}
    
    def __setitem__(self, key, value):
        self._data[key.lower()] = value
    
    def __getitem__(self, key):
        return self._data[key.lower()]
    
    def __contains__(self, key):
        return key.lower() in self._data
    
    def __iter__(self):
        return iter(self._data)

d = CaseInsensitiveDict()
d["Name"] = "John"
print(d["name"])  # "John" - case insensitive
print("NAME" in d)  # True
```

---

### Q8: How does Python's `*args` and `**kwargs` work, and when should you use them?

**Answer:**
`*args` collects positional arguments as a tuple, `**kwargs` collects keyword arguments as a dictionary.

```python
# Basic usage
def print_args(*args, **kwargs):
    print("Positional:", args)
    print("Keyword:", kwargs)

print_args(1, 2, 3, name="John", age=30)
# Positional: (1, 2, 3)
# Keyword: {'name': 'John', 'age': 30}

# Unpacking arguments
def add(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
print(add(*numbers))  # Unpacks list as arguments

params = {"a": 1, "b": 2, "c": 3}
print(add(**params))  # Unpacks dict as keyword arguments

# Combining with regular parameters
def greet(greeting, *names, **options):
    sep = options.get("sep", ", ")
    return f"{greeting} {sep.join(names)}!"

print(greet("Hello", "Alice", "Bob", sep=" and "))
# "Hello Alice and Bob!"

# Forwarding arguments to another function
def wrapper(*args, **kwargs):
    print("Before function call")
    result = original_function(*args, **kwargs)
    print("After function call")
    return result

def original_function(x, y, z=10):
    return x + y + z

# Decorator pattern
def logger(func):
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        result = func(*args, **kwargs)
        print(f"Result: {result}")
        return result
    return wrapper

@logger
def multiply(a, b):
    return a * b

multiply(3, 4)

# Enforcing keyword-only arguments (Python 3+)
def create_user(name, *, email, age):
    # email and age MUST be passed as keyword arguments
    return {"name": name, "email": email, "age": age}

# create_user("John", "john@example.com", 30)  # Error!
user = create_user("John", email="john@example.com", age=30)  # Correct

# Positional-only arguments (Python 3.8+)
def process(a, b, /, c, d, *, e, f):
    # a, b: positional-only
    # c, d: positional or keyword
    # e, f: keyword-only
    return a + b + c + d + e + f

# process(1, 2, 3, 4, 5, 6)  # Error - e, f must be keyword
result = process(1, 2, 3, 4, e=5, f=6)  # Correct

# Real-world example: flexible API client
class APIClient:
    def request(self, endpoint, method="GET", **params):
        url = f"https://api.example.com/{endpoint}"
        # params can include: headers, timeout, auth, etc.
        print(f"{method} {url}")
        for key, value in params.items():
            print(f"  {key}: {value}")

client = APIClient()
client.request("users", method="POST", 
               headers={"Authorization": "Bearer token"},
               timeout=30,
               data={"name": "John"})
```

---

### Q9: What is the difference between shallow and deep copy in Python?

**Answer:**
Shallow copy creates a new object but references nested objects, while deep copy recursively copies all nested objects.

```python
import copy

# Shallow copy
original = [[1, 2, 3], [4, 5, 6]]
shallow = copy.copy(original)  # or original.copy() or original[:]

shallow[0][0] = 999
print(original)  # [[999, 2, 3], [4, 5, 6]] - Modified!
print(shallow)   # [[999, 2, 3], [4, 5, 6]]

# Deep copy
original = [[1, 2, 3], [4, 5, 6]]
deep = copy.deepcopy(original)

deep[0][0] = 999
print(original)  # [[1, 2, 3], [4, 5, 6]] - Unchanged!
print(deep)      # [[999, 2, 3], [4, 5, 6]]

# Visualization of the difference
original = [1, 2, [3, 4]]

# Shallow copy - inner list is shared
shallow = original[:]
print(id(original))  # Different ID
print(id(shallow))
print(id(original[2]))  # Same ID - shared reference!
print(id(shallow[2]))

# Deep copy - everything is new
deep = copy.deepcopy(original)
print(id(deep[2]))  # Different ID - independent copy

# Dictionary example
original_dict = {
    "name": "John",
    "scores": [90, 85, 95],
    "metadata": {"created": "2024-01-01"}
}

shallow_dict = original_dict.copy()
shallow_dict["scores"].append(100)
print(original_dict["scores"])  # [90, 85, 95, 100] - Modified!

deep_dict = copy.deepcopy(original_dict)
deep_dict["scores"].append(80)
print(original_dict["scores"])  # Unchanged

# When to use each
# Shallow copy: when you only need to modify top-level items
user = {"name": "Alice", "age": 30}
user_copy = user.copy()
user_copy["name"] = "Bob"  # Safe - only top level modified

# Deep copy: when you have nested structures you want independent
class Node:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []

tree = Node(1, [Node(2), Node(3)])
tree_copy = copy.deepcopy(tree)  # Complete independent copy

# Performance consideration
import timeit

large_list = [[i] * 100 for i in range(1000)]

shallow_time = timeit.timeit(
    lambda: copy.copy(large_list),
    number=10000
)

deep_time = timeit.timeit(
    lambda: copy.deepcopy(large_list),
    number=10000
)

print(f"Shallow: {shallow_time:.4f}s")
print(f"Deep: {deep_time:.4f}s")
# Deep copy is significantly slower!
```

---

### Q10: How do you work with JSON data in Python effectively?

**Answer:**
Python's `json` module provides encoding/decoding with options for custom serialization.

```python
import json
from datetime import datetime
from decimal import Decimal

# Basic encoding and decoding
data = {
    "name": "John",
    "age": 30,
    "languages": ["Python", "JavaScript"],
    "active": True
}

# To JSON string
json_string = json.dumps(data, indent=2)
print(json_string)

# From JSON string
parsed = json.loads(json_string)

# File operations
with open("data.json", "w") as f:
    json.dump(data, f, indent=2)

with open("data.json", "r") as f:
    loaded = json.load(f)

# Custom serialization for non-JSON types
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        if isinstance(obj, Decimal):
            return float(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)

complex_data = {
    "timestamp": datetime.now(),
    "amount": Decimal("99.99"),
    "tags": {"python", "json", "tutorial"}
}

json_str = json.dumps(complex_data, cls=CustomEncoder, indent=2)
print(json_str)

# Custom deserialization
def datetime_decoder(dct):
    for key, value in dct.items():
        if key == "timestamp":
            dct[key] = datetime.fromisoformat(value)
    return dct

decoded = json.loads(json_str, object_hook=datetime_decoder)

# Handling nested JSON
nested = {
    "user": {
        "profile": {
            "name": "Alice",
            "settings": {
                "theme": "dark",
                "notifications": True
            }
        }
    }
}

# Safe access with get()
theme = nested.get("user", {}).get("profile", {}).get("settings", {}).get("theme")

# Better: using a helper function
def get_nested(data, *keys, default=None):
    for key in keys:
        if isinstance(data, dict):
            data = data.get(key)
        else:
            return default
    return data if data is not None else default

theme = get_nested(nested, "user", "profile", "settings", "theme")

# JSON Schema validation (requires jsonschema)
# pip install jsonschema
from jsonschema import validate, ValidationError

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "number", "minimum": 0}
    },
    "required": ["name", "age"]
}

valid_data = {"name": "John", "age": 30}
try:
    validate(instance=valid_data, schema=schema)
    print("Valid!")
except ValidationError as e:
    print(f"Invalid: {e.message}")

# Streaming large JSON files
def stream_json_array(file_path):
    """Stream large JSON array without loading everything in memory"""
    with open(file_path, 'r') as f:
        f.read(1)  # Skip opening bracket
        buffer = ""
        for line in f:
            buffer += line
            if line.strip().endswith('},'):
                yield json.loads(buffer.rstrip(','))
                buffer = ""

# Pretty printing
print(json.dumps(data, indent=2, sort_keys=True))

# Compact printing (for APIs)
compact = json.dumps(data, separators=(',', ':'))
```

---

### Q11: What are Python's context managers and how do you create custom ones?

**Answer:**
Context managers handle setup and teardown logic automatically using `with` statement.

```python
# Built-in context manager - file handling
with open("file.txt", "w") as f:
    f.write("Hello")
# File automatically closed, even if exception occurs

# Multiple context managers
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    content = infile.read()
    outfile.write(content.upper())

# Custom context manager - class-based
class DatabaseConnection:
    def __init__(self, connection_string):
        self.connection_string = connection_string
        self.connection = None
    
    def __enter__(self):
        print(f"Connecting to {self.connection_string}")
        self.connection = f"Connection to {self.connection_string}"
        return self.connection
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print("Closing connection")
        self.connection = None
        # Return False to propagate exceptions
        # Return True to suppress exceptions
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}")
        return False

with DatabaseConnection("localhost:5432") as conn:
    print(f"Using {conn}")
    # raise ValueError("Something went wrong")

# Custom context manager - function-based using contextlib
from contextlib import contextmanager

@contextmanager
def timer(label):
    import time
    start = time.time()
    try:
        yield  # Code block executes here
    finally:
        end = time.time()
        print(f"{label}: {end - start:.2f}s")

with timer("Processing"):
    import time
    time.sleep(1)
    print("Doing work")

# Reusable file handler with error logging
@contextmanager
def safe_file_handler(filename, mode='r'):
    f = None
    try:
        f = open(filename, mode)
        yield f
    except IOError as e:
        print(f"Error accessing file: {e}")
        yield None
    finally:
        if f:
            f.close()

with safe_file_handler("data.txt", "r") as f:
    if f:
        content = f.read()

# Lock context manager for thread safety
from contextlib import contextmanager
import threading

lock = threading.Lock()

@contextmanager
def synchronized(lock):
    lock.acquire()
    try:
        yield
    finally:
        lock.release()

# Usage
shared_resource = 0
with synchronized(lock):
    shared_resource += 1

# Temporary directory context manager
import tempfile
import shutil
import os

@contextmanager
def temporary_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

with temporary_directory() as temp_dir:
    # Create files in temp directory
    file_path = os.path.join(temp_dir, "temp_file.txt")
    with open(file_path, "w") as f:
        f.write("Temporary data")
# temp_dir automatically deleted

# Suppress exceptions context manager
from contextlib import suppress

# Instead of try-except
with suppress(FileNotFoundError):
    os.remove("nonexistent_file.txt")
# No error raised if file doesn't exist

# Chaining context managers
from contextlib import ExitStack

def process_files(filenames):
    with ExitStack() as stack:
        files = [stack.enter_context(open(fn)) for fn in filenames]
        # All files automatically closed when exiting
        for f in files:
            print(f.read())

# Real-world example: database transaction
@contextmanager
def transaction(connection):
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
        print("Transaction committed")
    except Exception as e:
        connection.rollback()
        print(f"Transaction rolled back: {e}")
        raise

# Usage:
# with transaction(db_connection) as cursor:
#     cursor.execute("INSERT INTO users VALUES (?)", ("John",))
```

---

### Q12: How does Python's slice notation work and what are advanced slicing techniques?

**Answer:**
Slicing uses `[start:stop:step]` syntax with powerful features for sequence manipulation.

```python
# Basic slicing
numbers = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

print(numbers[2:7])     # [2, 3, 4, 5, 6] - start to stop-1
print(numbers[:5])      # [0, 1, 2, 3, 4] - from beginning
print(numbers[5:])      # [5, 6, 7, 8, 9] - to end
print(numbers[:])       # Full copy of list

# Step parameter
print(numbers[::2])     # [0, 2, 4, 6, 8] - every 2nd element
print(numbers[1::2])    # [1, 3, 5, 7, 9] - odd indices
print(numbers[::3])     # [0, 3, 6, 9] - every 3rd element

# Negative indices
print(numbers[-3:])     # [7, 8, 9] - last 3 elements
print(numbers[:-3])     # [0, 1, 2, 3, 4, 5, 6] - except last 3
print(numbers[-5:-2])   # [5, 6, 7] - range from end

# Reverse sequence
print(numbers[::-1])    # [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
print(numbers[::-2])    # [9, 7, 5, 3, 1] - reverse, every 2nd

# Slice assignment
numbers = [0, 1, 2, 3, 4, 5]
numbers[2:4] = [20, 30, 40]  # Replace with different length
print(numbers)  # [0, 1, 20, 30, 40, 4, 5]

numbers[1:4] = []  # Delete elements
print(numbers)  # [0, 40, 4, 5]

# Insert elements
numbers = [0, 1, 2, 3, 4]
numbers[2:2] = [10, 20]  # Insert at index 2
print(numbers)  # [0, 1, 10, 20, 2, 3, 4]

# Slice object for reusability
LAST_THREE = slice(-3, None)
EVEN_INDICES = slice(None, None, 2)

data = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
print(data[LAST_THREE])   # [7, 8, 9]
print(data[EVEN_INDICES]) # [0, 2, 4, 6, 8]

# Named slices improve readability
record = "john:doe:30:engineer:boston"
fields = record.split(':')

# Instead of magic numbers:
# name = fields[0] + ' ' + fields[1]
# age = fields[2]
# job = fields[3]

# Use named slices:
FIRST_NAME = slice(0, 1)
LAST_NAME = slice(1, 2)
AGE = slice(2, 3)
JOB = slice(3, 4)

first_name = fields[FIRST_NAME][0]
last_name = fields[LAST_NAME][0]

# String slicing
text = "Python Programming"
print(text[0:6])        # "Python"
print(text[7:])         # "Programming"
print(text[::-1])       # "gnimmargorP nohtyP" - reverse

# Check if string is palindrome
word = "racecar"
print(word == word[::-1])  # True

# Extract file extension
filename = "document.pdf"
extension = filename[filename.rfind('.'):]
# Or better:
extension = filename.split('.')[-1]

# Multi-dimensional slicing (NumPy-style with lists)
matrix = [
    [1, 2, 3, 4],
    [5, 6, 7, 8],
    [9, 10, 11, 12]
]

# Get all rows, columns 1-2
result = [row[1:3] for row in matrix]
print(result)  # [[2, 3], [6, 7], [10, 11]]

# Practical examples

# 1. Pagination
def paginate(items, page_size=10, page_number=1):
    start = (page_number - 1) * page_size
    end = start + page_size
    return items[start:end]

all_items = list(range(100))
page_1 = paginate(all_items, page_size=10, page_number=1)
print(page_1)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# 2. Rotating a list
def rotate_left(lst, n):
    n = n % len(lst)  # Handle n > len(lst)
    return lst[n:] + lst[:n]

numbers = [1, 2, 3, 4, 5]
print(rotate_left(numbers, 2))  # [3, 4, 5, 1, 2]

# 3. Chunk data into batches
def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

data = list(range(10))
for chunk in chunks(data, 3):
    print(chunk)
# [0, 1, 2]
# [3, 4, 5]
# [6, 7, 8]
# [9]

# 4. Window sliding
def sliding_window(lst, window_size):
    for i in range(len(lst) - window_size + 1):
        yield lst[i:i + window_size]

numbers = [1, 2, 3, 4, 5]
for window in sliding_window(numbers, 3):
    print(window)
# [1, 2, 3]
# [2, 3, 4]
# [3, 4, 5]
```

---

### Q13: What are list comprehensions and how do they compare to map/filter?

**Answer:**
List comprehensions provide concise syntax for creating lists and are often more readable than map/filter.

```python
# Basic list comprehension
squares = [x**2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# Equivalent using map
squares_map = list(map(lambda x: x**2, range(10)))

# With conditional (filter)
even_squares = [x**2 for x in range(10) if x % 2 == 0]
print(even_squares)  # [0, 4, 16, 36, 64]

# Equivalent using filter + map
even_squares_functional = list(map(lambda x: x**2, filter(lambda x: x % 2 == 0, range(10))))

# Multiple conditions
result = [x for x in range(20) if x % 2 == 0 if x % 3 == 0]
print(result)  # [0, 6, 12, 18]

# if-else in comprehension (ternary operator)
values = [x if x % 2 == 0 else -x for x in range(10)]
print(values)  # [0, -1, 2, -3, 4, -5, 6, -7, 8, -9]

# Nested loops
matrix = [[i * j for j in range(3)] for i in range(3)]
print(matrix)
# [[0, 0, 0], [0, 1, 2], [0, 2, 4]]

# Flatten nested list
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flattened = [item for sublist in nested for item in sublist]
print(flattened)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Set comprehension (removes duplicates)
text = "hello world"
unique_chars = {char.lower() for char in text if char.isalpha()}
print(unique_chars)  # {'h', 'e', 'l', 'o', 'w', 'r', 'd'}

# Dictionary comprehension
word_lengths = {word: len(word) for word in ["hello", "world", "python"]}
print(word_lengths)  # {'hello': 5, 'world': 5, 'python': 6}

# Generator expression (memory efficient)
sum_squares = sum(x**2 for x in range(1000000))  # No list created!

# Real-world examples

# 1. Data transformation
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]

names = [user["name"] for user in users]
adults = [user for user in users if user["age"] >= 30]

# 2. CSV processing
csv_lines = ["name,age,city", "Alice,30,NYC", "Bob,25,LA"]
data = [line.split(',') for line in csv_lines[1:]]  # Skip header
print(data)  # [['Alice', '30', 'NYC'], ['Bob', '25', 'LA']]

# 3. File processing
# filenames = [f for f in os.listdir('.') if f.endswith('.py')]

# 4. String manipulation
sentence = "Hello World Python"
words_upper = [word.upper() for word in sentence.split()]
print(words_upper)  # ['HELLO', 'WORLD', 'PYTHON']

# Performance comparison
import timeit

# List comprehension
def list_comp():
    return [x**2 for x in range(1000)]

# Map
def map_func():
    return list(map(lambda x: x**2, range(1000)))

# Traditional loop
def for_loop():
    result = []
    for x in range(1000):
        result.append(x**2)
    return result

print("List comp:", timeit.timeit(list_comp, number=10000))
print("Map:", timeit.timeit(map_func, number=10000))
print("For loop:", timeit.timeit(for_loop, number=10000))
# List comprehension is typically fastest!

# When NOT to use comprehensions
# Too complex - hurts readability
# Bad:
# result = [[cell.upper() if cell.isalpha() else cell.lower() 
#           for cell in row if len(cell) > 3] 
#          for row in matrix if sum(len(c) for c in row) > 10]

# Better as traditional loop with clear logic

# Walrus operator in comprehensions (Python 3.8+)
data = [1, 2, 3, 4, 5]
# Calculate expensive operation once
result = [y for x in data if (y := x**2) > 10]
print(result)  # [16, 25]
```

---

### Q14: How do you handle exceptions properly in Python?

**Answer:**
Use specific exception types, proper exception hierarchy, and clean error handling patterns.

```python
# Basic exception handling
try:
    result = 10 / 0
except ZeroDivisionError as e:
    print(f"Error: {e}")

# Multiple exception types
try:
    value = int(input("Enter a number: "))
    result = 100 / value
except ValueError:
    print("Invalid number format")
except ZeroDivisionError:
    print("Cannot divide by zero")
except Exception as e:
    print(f"Unexpected error: {e}")

# else clause - runs if no exception
try:
    file = open("data.txt", "r")
except FileNotFoundError:
    print("File not found")
else:
    # Only runs if no exception
    content = file.read()
    file.close()
    print("File read successfully")

# finally clause - always runs
try:
    connection = create_connection()
    data = connection.fetch_data()
except ConnectionError:
    print("Connection failed")
finally:
    # Always runs, even if exception occurs
    if 'connection' in locals():
        connection.close()

# Complete pattern
def process_file(filename):
    file = None
    try:
        file = open(filename, 'r')
        data = file.read()
        return process_data(data)
    except FileNotFoundError:
        print(f"File {filename} not found")
        return None
    except PermissionError:
        print(f"No permission to read {filename}")
        return None
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None
    else:
        print("File processed successfully")
    finally:
        if file:
            file.close()

# Custom exceptions
class ValidationError(Exception):
    """Raised when data validation fails"""
    pass

class InsufficientFundsError(Exception):
    """Raised when account has insufficient funds"""
    def __init__(self, balance, amount):
        self.balance = balance
        self.amount = amount
        super().__init__(f"Insufficient funds: balance={balance}, required={amount}")

def withdraw(account, amount):
    if amount > account.balance:
        raise InsufficientFundsError(account.balance, amount)
    account.balance -= amount

# Using custom exceptions
class Account:
    def __init__(self, balance):
        self.balance = balance

account = Account(100)
try:
    withdraw(account, 150)
except InsufficientFundsError as e:
    print(f"Transaction failed: {e}")
    print(f"Current balance: ${e.balance}")
    print(f"Attempted withdrawal: ${e.amount}")

# Exception chaining
def load_config(filename):
    try:
        with open(filename) as f:
            return json.load(f)
    except FileNotFoundError as e:
        raise ConfigError(f"Config file {filename} not found") from e
    except json.JSONDecodeError as e:
        raise ConfigError(f"Invalid JSON in {filename}") from e

# Catching multiple exceptions as one
try:
    # some code
    pass
except (TypeError, ValueError, KeyError) as e:
    print(f"Input error: {e}")

# Context manager for exception handling
from contextlib import suppress

# Ignore specific exceptions
with suppress(FileNotFoundError):
    os.remove("temp_file.txt")

# Re-raising exceptions
def validate_age(age):
    try:
        age = int(age)
        if age < 0:
            raise ValueError("Age cannot be negative")
        return age
    except ValueError:
        print("Validation failed")
        raise  # Re-raise the same exception

# Exception groups (Python 3.11+)
# try:
#     # code that might raise multiple exceptions
#     pass
# except* ValueError as eg:
#     for e in eg.exceptions:
#         print(f"ValueError: {e}")
# except* TypeError as eg:
#     for e in eg.exceptions:
#         print(f"TypeError: {e}")

# Best practices

# 1. Be specific with exceptions
# Bad:
try:
    result = risky_operation()
except Exception:  # Too broad!
    pass

# Good:
try:
    result = risky_operation()
except (ConnectionError, TimeoutError) as e:
    handle_network_error(e)

# 2. Don't use bare except
# Bad:
try:
    something()
except:  # Catches everything, including KeyboardInterrupt!
    pass

# Good:
try:
    something()
except Exception as e:  # Doesn't catch system exits
    log_error(e)

# 3. Create exception hierarchy
class DatabaseError(Exception):
    """Base class for database exceptions"""
    pass

class ConnectionError(DatabaseError):
    """Database connection failed"""
    pass

class QueryError(DatabaseError):
    """Query execution failed"""
    pass

# Catch specific or general
try:
    execute_query()
except QueryError:
    # Handle query issues
    pass
except DatabaseError:
    # Handle all other database issues
    pass

# 4. Logging exceptions
import logging

try:
    risky_operation()
except Exception:
    logging.exception("Operation failed")  # Logs full traceback

# 5. Type hints for exceptions (Python 3.11+)
def divide(a: int, b: int) -> float:
    """Divide two numbers.
    
    Raises:
        ZeroDivisionError: If b is zero
        TypeError: If arguments are not numbers
    """
    if b == 0:
        raise ZeroDivisionError("Cannot divide by zero")
    return a / b
```

---

### Q15: What are Python's built-in functions you should know?

**Answer:**
Python has 69 built-in functions providing essential operations without imports.

```python
# Type conversions
print(int("42"))        # 42
print(float("3.14"))    # 3.14
print(str(123))         # "123"
print(bool(0))          # False
print(list("hello"))    # ['h', 'e', 'l', 'l', 'o']
print(tuple([1, 2, 3])) # (1, 2, 3)
print(set([1, 2, 2, 3])) # {1, 2, 3}

# Numeric operations
print(abs(-42))         # 42
print(pow(2, 10))       # 1024 (same as 2**10)
print(round(3.14159, 2)) # 3.14
print(divmod(17, 5))    # (3, 2) - quotient and remainder
print(sum([1, 2, 3, 4])) # 10
print(min([5, 2, 8, 1])) # 1
print(max([5, 2, 8, 1])) # 8

# Advanced min/max with key
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78}
]
best_student = max(students, key=lambda s: s["grade"])
print(best_student)  # {'name': 'Bob', 'grade': 92}

# Sequence operations
numbers = [1, 2, 3, 4, 5]
print(len(numbers))     # 5
print(reversed(numbers)) # reversed object (iterator)
print(list(reversed(numbers))) # [5, 4, 3, 2, 1]

# sorted() - creates new sorted list
unsorted = [3, 1, 4, 1, 5, 9, 2]
print(sorted(unsorted))  # [1, 1, 2, 3, 4, 5, 9]
print(sorted(unsorted, reverse=True))  # [9, 5, 4, 3, 2, 1, 1]

# Sort by custom key
words = ["banana", "pie", "Washington", "book"]
print(sorted(words, key=len))  # ['pie', 'book', 'banana', 'Washington']
print(sorted(words, key=str.lower))  # Case-insensitive sort

# enumerate - get index and value
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
# 0: apple
# 1: banana
# 2: cherry

# Start enumerate from different number
for index, fruit in enumerate(fruits, start=1):
    print(f"{index}: {fruit}")

# zip - combine multiple iterables
names = ["Alice", "Bob", "Charlie"]
ages = [30, 25, 35]
cities = ["NYC", "LA", "Chicago"]

for name, age, city in zip(names, ages, cities):
    print(f"{name} is {age} years old and lives in {city}")

# zip stops at shortest sequence
short = [1, 2]
long = [10, 20, 30, 40]
print(list(zip(short, long)))  # [(1, 10), (2, 20)]

# Create dictionary from two lists
keys = ["name", "age", "city"]
values = ["John", 30, "Boston"]
person = dict(zip(keys, values))
print(person)

# map - apply function to each element
numbers = [1, 2, 3, 4, 5]
squared = map(lambda x: x**2, numbers)
print(list(squared))  # [1, 4, 9, 16, 25]

# map with multiple sequences
a = [1, 2, 3]
b = [10, 20, 30]
result = map(lambda x, y: x + y, a, b)
print(list(result))  # [11, 22, 33]

# filter - keep elements that pass test
numbers = range(10)
evens = filter(lambda x: x % 2 == 0, numbers)
print(list(evens))  # [0, 2, 4, 6, 8]

# any - True if any element is truthy
print(any([False, False, True, False]))  # True
print(any([0, [], "", None]))  # False

# Check if any number is negative
numbers = [1, 5, -3, 7]
has_negative = any(n < 0 for n in numbers)
print(has_negative)  # True

# all - True if all elements are truthy
print(all([True, True, True]))  # True
print(all([True, False, True]))  # False

# Check if all numbers are positive
numbers = [1, 5, 3, 7]
all_positive = all(n > 0 for n in numbers)
print(all_positive)  # True

# range - generate sequence of numbers
print(list(range(5)))        # [0, 1, 2, 3, 4]
print(list(range(2, 10)))    # [2, 3, 4, 5, 6, 7, 8, 9]
print(list(range(0, 10, 2))) # [0, 2, 4, 6, 8]

# Introspection functions
class MyClass:
    def __init__(self):
        self.x = 10

obj = MyClass()

print(type(obj))           # <class '__main__.MyClass'>
print(isinstance(obj, MyClass))  # True
print(hasattr(obj, 'x'))   # True
print(getattr(obj, 'x'))   # 10
setattr(obj, 'y', 20)
print(obj.y)               # 20
print(dir(obj))            # List all attributes

# id - unique identifier of object
a = [1, 2, 3]
b = [1, 2, 3]
print(id(a))  # Different from id(b)
print(a is b)  # False

# callable - check if object can be called
def func():
    pass

print(callable(func))      # True
print(callable(42))        # False
print(callable(list))      # True

# input/output
name = input("Enter name: ")  # Read from stdin
print("Hello", name)          # Write to stdout

# format numbers
print(format(1234, ','))      # "1,234"
print(format(0.123, '.2%'))   # "12.30%"

# bin, oct, hex - number conversions
print(bin(10))   # "0b1010"
print(oct(10))   # "0o12"
print(hex(255))  # "0xff"

# chr, ord - character conversions
print(chr(65))   # "A"
print(ord('A'))  # 65

# Complex number operations
c = complex(3, 4)  # 3 + 4j
print(abs(c))      # 5.0 - magnitude

# iter and next - manual iteration
my_list = [1, 2, 3]
iterator = iter(my_list)
print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
# print(next(iterator))  # StopIteration error

# With default value
print(next(iterator, "Done"))  # "Done" instead of error

# eval - evaluate string as code (use carefully!)
result = eval("2 + 3 * 4")
print(result)  # 14

# exec - execute string as code
code = """
def greet(name):
    return f"Hello, {name}"
    
print(greet("World"))
"""
exec(code)

# compile - compile code for later execution
code = compile("print('Hello')", "<string>", "exec")
exec(code)

# vars() - return __dict__ of object
print(vars(obj))  # {'x': 10, 'y': 20}

# globals() and locals()
def func():
    local_var = 42
    print(locals())  # Shows local variables
    print(globals())  # Shows global variables

# help - interactive documentation
help(sorted)  # Shows documentation for sorted()
```

---

### Q16: What are Python's string formatting methods and when to use each?

**Answer:**
Python offers %-formatting, str.format(), and f-strings (preferred in modern code).

```python
# 1. f-strings (Python 3.6+) - PREFERRED METHOD
name = "Alice"
age = 30
city = "NYC"

# Basic f-string
message = f"Hello, {name}!"
print(message)  # "Hello, Alice!"

# Expressions in f-strings
print(f"{name} will be {age + 1} next year")

# Formatting numbers
price = 19.99
print(f"Price: ${price:.2f}")  # "Price: $19.99"

# Alignment and padding
print(f"{name:>10}")   # "     Alice" - right aligned
print(f"{name:<10}")   # "Alice     " - left aligned
print(f"{name:^10}")   # "  Alice   " - centered

# Numbers with separators
large_num = 1000000
print(f"{large_num:,}")  # "1,000,000"

# Percentage
ratio = 0.857
print(f"{ratio:.1%}")  # "85.7%"

# Binary, octal, hex
num = 42
print(f"{num:b}")  # "101010" - binary
print(f"{num:o}")  # "52" - octal
print(f"{num:x}")  # "2a" - hex
print(f"{num:X}")  # "2A" - hex uppercase

# Datetime formatting
from datetime import datetime
now = datetime.now()
print(f"{now:%Y-%m-%d %H:%M:%S}")  # "2024-01-15 14:30:45"

# Debug f-strings (Python 3.8+)
x = 42
y = 10
print(f"{x=}, {y=}")  # "x=42, y=10"
print(f"{x + y=}")    # "x + y=52"

# 2. str.format() - Still useful for templates
template = "Hello, {}! You are {} years old."
message = template.format(name, age)
print(message)

# Named placeholders
template = "Hello, {name}! You live in {city}."
message = template.format(name=name, city=city)

# Positional and keyword combined
template = "{0} is {1} years old and lives in {city}"
message = template.format(name, age, city=city)

# Format specifications
print("{:.2f}".format(3.14159))  # "3.14"
print("{:,}".format(1000000))    # "1,000,000"
print("{:>10}".format("right"))  # "     right"

# Reusing arguments
template = "{0} loves {1}. {0} really loves {1}!"
print(template.format("Alice", "Python"))

# Dictionary unpacking
person = {"name": "Bob", "age": 25}
print("{name} is {age} years old".format(**person))

# 3. % formatting (old style - avoid in new code)
print("Hello, %s!" % name)
print("%s is %d years old" % (name, age))
print("Price: $%.2f" % price)

# Multiple values
print("Name: %s, Age: %d, City: %s" % (name, age, city))

# Dictionary formatting
print("%(name)s is %(age)d" % {"name": name, "age": age})

# Real-world examples

# 1. Logging with timestamps
def log(message):
    timestamp = datetime.now()
    print(f"[{timestamp:%Y-%m-%d %H:%M:%S}] {message}")

log("Application started")

# 2. Table formatting
def print_table(data):
    # Data: list of dictionaries
    headers = data[0].keys()
    
    # Calculate column widths
    widths = {h: max(len(str(h)), max(len(str(row[h])) for row in data)) 
              for h in headers}
    
    # Print header
    header_line = " | ".join(f"{h:<{widths[h]}}" for h in headers)
    print(header_line)
    print("-" * len(header_line))
    
    # Print rows
    for row in data:
        print(" | ".join(f"{str(row[h]):<{widths[h]}}" for h in headers))

users = [
    {"name": "Alice", "age": 30, "city": "NYC"},
    {"name": "Bob", "age": 25, "city": "LA"},
    {"name": "Charlie", "age": 35, "city": "Chicago"}
]
print_table(users)

# 3. Progress bar
def progress_bar(current, total, width=50):
    percent = current / total
    filled = int(width * percent)
    bar = "█" * filled + "░" * (width - filled)
    print(f"\r[{bar}] {percent:.1%}", end="", flush=True)

# 4. File size formatting
def format_bytes(bytes):
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0

print(format_bytes(1234567))  # "1.18 MB"

# 5. SQL query building (be careful of SQL injection!)
# Safe way - use parameterized queries
query_template = "SELECT * FROM users WHERE age > {min_age} AND city = {city}"
# Don't use: query = query_template.format(min_age=25, city=user_input)
# Use parameterized queries with your DB library instead

# 6. API response formatting
def format_api_response(status, data, elapsed_time):
    return f"""
    Status: {status}
    Response Time: {elapsed_time:.3f}s
    Data: {data}
    """

# 7. Currency formatting
def format_currency(amount, currency="USD"):
    symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
    symbol = symbols.get(currency, currency)
    return f"{symbol}{amount:,.2f}"

print(format_currency(1234.56))  # "$1,234.56"

# 8. Multiline f-strings
report = f"""
Sales Report
============
Date: {datetime.now():%Y-%m-%d}
Total Sales: ${12345.67:,.2f}
Number of Transactions: {42}
Average: ${12345.67/42:,.2f}
"""
print(report)
```

---

### Q17: How do Python's comparison and identity operators work?

**Answer:**
Comparison operators compare values, while identity operators check if objects are the same instance.

```python
# Comparison operators: ==, !=, <, >, <=, >=
print(5 == 5)     # True
print(5 != 3)     # True
print(5 > 3)      # True
print(5 >= 5)     # True

# Identity operators: is, is not
a = [1, 2, 3]
b = [1, 2, 3]
c = a

print(a == b)     # True - same values
print(a is b)     # False - different objects
print(a is c)     # True - same object

# Check identity
print(id(a))      # Memory address
print(id(b))      # Different address
print(id(c))      # Same as a

# Integer caching (-5 to 256)
x = 256
y = 256
print(x is y)     # True - cached

x = 257
y = 257
print(x is y)     # May be False - not cached

# None comparison - always use 'is'
value = None
if value is None:  # Correct
    print("Value is None")

if value == None:  # Works but not idiomatic
    print("Value is None")

# Boolean comparison
flag = True
if flag is True:   # Don't do this
    pass

if flag:           # Do this instead
    pass

# String interning
s1 = "hello"
s2 = "hello"
print(s1 is s2)    # True - strings are interned

s1 = "hello world"
s2 = "hello world"
print(s1 is s2)    # May be False - not interned

# Chained comparisons
x = 5
print(1 < x < 10)  # True - cleaner than: 1 < x and x < 10
print(10 > x >= 5) # True

# Custom comparison in classes
class Version:
    def __init__(self, major, minor):
        self.major = major
        self.minor = minor
    
    def __eq__(self, other):
        return self.major == other.major and self.minor == other.minor
    
    def __lt__(self, other):
        if self.major != other.major:
            return self.major < other.major
        return self.minor < other.minor
    
    def __le__(self, other):
        return self == other or self < other
    
    # __gt__, __ge__, __ne__ are automatically derived

v1 = Version(1, 0)
v2 = Version(2, 0)
v3 = Version(1, 5)

print(v1 < v2)     # True
print(v1 < v3)     # True
print(v2 > v3)     # True

# Membership operators: in, not in
fruits = ["apple", "banana", "cherry"]
print("apple" in fruits)      # True
print("grape" not in fruits)  # True

# Works with strings
text = "hello world"
print("world" in text)         # True
print("xyz" not in text)       # True

# Works with dictionaries (checks keys)
person = {"name": "Alice", "age": 30}
print("name" in person)        # True
print("Alice" in person)       # False - checks keys only
print("Alice" in person.values())  # True

# Logical operators: and, or, not
print(True and True)   # True
print(True or False)   # True
print(not True)        # False

# Short-circuit evaluation
def expensive_check():
    print("Expensive check called")
    return True

# 'and' short-circuits if first is False
if False and expensive_check():
    pass  # expensive_check() never called

# 'or' short-circuits if first is True
if True or expensive_check():
    pass  # expensive_check() never called

# Truthy and falsy values
# Falsy: False, None, 0, 0.0, '', [], {}, set()
# Everything else is truthy

if []:
    print("Not printed")  # Empty list is falsy

if [1, 2, 3]:
    print("Printed")      # Non-empty list is truthy

# Using 'or' for default values
name = input("Enter name: ") or "Anonymous"
# If empty string, uses "Anonymous"

# Using 'and' for conditional execution
user_logged_in = True
premium_user = True
user_logged_in and premium_user and show_premium_content()

# Comparison of different types
# Python 3 doesn't allow comparing incompatible types
# print(5 < "hello")  # TypeError in Python 3

# But None can be compared
print(None == None)    # True
print(None is None)    # True (preferred)

# Custom ordering with key functions
names = ["alice", "Bob", "CHARLIE"]
print(sorted(names))              # Case-sensitive
print(sorted(names, key=str.lower))  # Case-insensitive

# Complex sorting
students = [
    {"name": "Alice", "grade": 85, "age": 20},
    {"name": "Bob", "grade": 92, "age": 19},
    {"name": "Charlie", "grade": 85, "age": 21}
]

# Sort by grade, then by age
sorted_students = sorted(students, key=lambda s: (s["grade"], s["age"]))

# Reverse sorting
sorted_desc = sorted(students, key=lambda s: s["grade"], reverse=True)

---

### Q18: What are Python decorators and how do you create them?

**Answer:**
Decorators modify or enhance functions/classes without changing their code, using the @decorator syntax.

```python
# Simple decorator
def simple_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@simple_decorator
def say_hello():
    print("Hello!")

say_hello()
# Before function
# Hello!
# After function

# Decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
# Hello, Alice!
# Hello, Alice!
# Hello, Alice!

# Preserving function metadata
from functools import wraps

def my_decorator(func):
    @wraps(func)  # Preserves original function's name, docstring, etc.
    def wrapper(*args, **kwargs):
        print(f"Calling {func.__name__}")
        return func(*args, **kwargs)
    return wrapper

@my_decorator
def example():
    """Example function"""
    pass

print(example.__name__)  # "example" (not "wrapper")
print(example.__doc__)   # "Example function"

# Timer decorator
import time

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def slow_function():
    time.sleep(1)
    return "Done"

slow_function()

# Caching decorator
def memoize(func):
    cache = {}
    @wraps(func)
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memoize
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))  # Fast due to caching!

# Better: use built-in lru_cache
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n < 2:
        return n
    return fibonacci_cached(n-1) + fibonacci_cached(n-2)

# Retry decorator
def retry(max_attempts=3, delay=1):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    if attempts == max_attempts:
                        raise
                    print(f"Attempt {attempts} failed: {e}")
                    time.sleep(delay)
        return wrapper
    return decorator

@retry(max_attempts=3, delay=2)
def unreliable_api_call():
    import random
    if random.random() < 0.7:
        raise ConnectionError("API unavailable")
    return "Success"

# Authentication decorator
def require_auth(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        user = kwargs.get('user')
        if not user or not user.get('authenticated'):
            raise PermissionError("Authentication required")
        return func(*args, **kwargs)
    return wrapper

@require_auth
def access_resource(resource_id, user=None):
    return f"Accessing resource {resource_id}"

# access_resource(123)  # PermissionError
access_resource(123, user={'authenticated': True})  # Works

# Validation decorator
def validate_types(**expected_types):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Validate argument types
            for arg_name, expected_type in expected_types.items():
                if arg_name in kwargs:
                    if not isinstance(kwargs[arg_name], expected_type):
                        raise TypeError(
                            f"{arg_name} must be {expected_type.__name__}"
                        )
            return func(*args, **kwargs)
        return wrapper
    return decorator

@validate_types(name=str, age=int)
def create_user(name, age):
    return {"name": name, "age": age}

user = create_user(name="Alice", age=30)  # Works
# user = create_user(name="Alice", age="30")  # TypeError

# Class decorator
def singleton(cls):
    instances = {}
    @wraps(cls)
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper

@singleton
class Database:
    def __init__(self):
        print("Creating database connection")

db1 = Database()  # "Creating database connection"
db2 = Database()  # No message - returns same instance
print(db1 is db2)  # True

# Method decorators
class MyClass:
    @staticmethod
    def static_method():
        print("Static method")
    
    @classmethod
    def class_method(cls):
        print(f"Class method of {cls.__name__}")
    
    @property
    def read_only(self):
        return self._value
    
    @read_only.setter
    def read_only(self, value):
        self._value = value

# Stacking decorators (applied bottom-up)
@timer
@memoize
def complex_calculation(n):
    return sum(range(n))

# Equivalent to:
# complex_calculation = timer(memoize(complex_calculation))

# Debug decorator
def debug(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        args_repr = [repr(a) for a in args]
        kwargs_repr = [f"{k}={v!r}" for k, v in kwargs.items()]
        signature = ", ".join(args_repr + kwargs_repr)
        print(f"Calling {func.__name__}({signature})")
        result = func(*args, **kwargs)
        print(f"{func.__name__} returned {result!r}")
        return result
    return wrapper

@debug
def add(a, b):
    return a + b

add(5, 3)
# Calling add(5, 3)
# add returned 8

# Rate limiting decorator
from collections import deque
from time import time

def rate_limit(max_calls, time_window):
    calls = deque()
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            now = time()
            # Remove old calls outside time window
            while calls and calls[0] < now - time_window:
                calls.popleft()
            
            if len(calls) >= max_calls:
                raise Exception(f"Rate limit exceeded: {max_calls} calls per {time_window}s")
            
            calls.append(now)
            return func(*args, **kwargs)
        return wrapper
    return decorator

@rate_limit(max_calls=5, time_window=60)
def api_call():
    return "API response"

---

### Q19: What is the difference between `@staticmethod`, `@classmethod`, and instance methods?

**Answer:**
These three method types serve different purposes in object-oriented programming.

```python
class MyClass:
    class_variable = "I'm a class variable"
    
    def __init__(self, value):
        self.instance_variable = value
    
    # Instance method - has access to instance (self)
    def instance_method(self):
        return f"Instance method can access: {self.instance_variable}"
    
    # Class method - has access to class (cls), not instance
    @classmethod
    def class_method(cls):
        return f"Class method can access: {cls.class_variable}"
    
    # Static method - no access to class or instance
    @staticmethod
    def static_method():
        return "Static method is like a regular function"

# Instance method usage
obj = MyClass("instance value")
print(obj.instance_method())
# "Instance method can access: instance value"

# Class method usage - can be called on class or instance
print(MyClass.class_method())
# "Class method can access: I'm a class variable"
print(obj.class_method())  # Also works

# Static method usage
print(MyClass.static_method())
print(obj.static_method())  # Also works

# Real-world use cases

# 1. Alternative constructors with @classmethod
class Date:
    def __init__(self, year, month, day):
        self.year = year
        self.month = month
        self.day = day
    
    @classmethod
    def from_string(cls, date_string):
        year, month, day = map(int, date_string.split('-'))
        return cls(year, month, day)
    
    @classmethod
    def today(cls):
        import datetime
        today = datetime.date.today()
        return cls(today.year, today.month, today.day)
    
    def __repr__(self):
        return f"Date({self.year}, {self.month}, {self.day})"

# Multiple ways to create Date
date1 = Date(2024, 1, 15)
date2 = Date.from_string("2024-01-15")
date3 = Date.today()

print(date1)
print(date2)
print(date3)

# 2. Factory pattern with @classmethod
class Pizza:
    def __init__(self, ingredients):
        self.ingredients = ingredients
    
    @classmethod
    def margherita(cls):
        return cls(["mozzarella", "tomatoes", "basil"])
    
    @classmethod
    def pepperoni(cls):
        return cls(["mozzarella", "tomatoes", "pepperoni"])
    
    def __repr__(self):
        return f"Pizza with {', '.join(self.ingredients)}"

pizza1 = Pizza.margherita()
pizza2 = Pizza.pepperoni()
print(pizza1)
print(pizza2)

# 3. Utility functions with @staticmethod
class StringUtils:
    @staticmethod
    def reverse(string):
        return string[::-1]
    
    @staticmethod
    def is_palindrome(string):
        clean = string.lower().replace(" ", "")
        return clean == clean[::-1]
    
    @staticmethod
    def truncate(string, length):
        return string[:length] + "..." if len(string) > length else string

print(StringUtils.reverse("hello"))
print(StringUtils.is_palindrome("racecar"))
print(StringUtils.truncate("Long text here", 8))

# 4. Counting instances with @classmethod
class Employee:
    num_employees = 0
    
    def __init__(self, name):
        self.name = name
        Employee.num_employees += 1
    
    @classmethod
    def get_employee_count(cls):
        return cls.num_employees
    
    @classmethod
    def from_string(cls, emp_string):
        name = emp_string.split('-')[0]
        return cls(name)

emp1 = Employee("Alice")
emp2 = Employee("Bob")
emp3 = Employee.from_string("Charlie-Developer")

print(Employee.get_employee_count())  # 3

# 5. Inheritance behavior
class Animal:
    species_count = 0
    
    @classmethod
    def add_species(cls):
        cls.species_count += 1
        return cls.species_count
    
    @staticmethod
    def make_sound():
        return "Some generic sound"

class Dog(Animal):
    @staticmethod
    def make_sound():
        return "Woof!"

class Cat(Animal):
    @staticmethod
    def make_sound():
        return "Meow!"

# Class method uses the actual class
print(Dog.add_species())    # Modifies Dog.species_count
print(Cat.add_species())    # Modifies Cat.species_count
print(Animal.species_count)  # 0 - unchanged

# Static methods can be overridden
print(Animal.make_sound())  # "Some generic sound"
print(Dog.make_sound())     # "Woof!"
print(Cat.make_sound())     # "Meow!"

# 6. Validation with @staticmethod
class Validator:
    @staticmethod
    def validate_email(email):
        import re
        pattern = r'^[\w\.-]+@[\w\.-]+\.\w+
        return bool(re.match(pattern, email))
    
    @staticmethod
    def validate_phone(phone):
        import re
        pattern = r'^\d{3}-\d{3}-\d{4}
        return bool(re.match(pattern, phone))

print(Validator.validate_email("user@example.com"))  # True
print(Validator.validate_phone("123-456-7890"))      # True

# When to use each:
# - Instance method: When you need access to instance data
# - Class method: Alternative constructors, factory methods, class-level operations
# - Static method: Utility functions related to the class but not needing class/instance data

---

### Q20: How does Python's `with` statement work and what are context managers?

**Answer:**
The `with` statement ensures proper resource cleanup using context managers that implement `__enter__` and `__exit__`.

```python
# Basic file handling without context manager (old way)
file = open("data.txt", "r")
try:
    content = file.read()
finally:
    file.close()  # Must remember to close

# With context manager (better)
with open("data.txt", "r") as file:
    content = file.read()
# File automatically closed, even if exception occurs

# Multiple context managers
with open("input.txt", "r") as infile, open("output.txt", "w") as outfile:
    content = infile.read()
    outfile.write(content.upper())

# Creating custom context manager - class-based
class FileManager:
    def __init__(self, filename, mode):
        self.filename = filename
        self.mode = mode
        self.file = None
    
    def __enter__(self):
        print(f"Opening {self.filename}")
        self.file = open(self.filename, self.mode)
        return self.file
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Closing {self.filename}")
        if self.file:
            self.file.close()
        # Return False to propagate exceptions
        # Return True to suppress exceptions
        if exc_type is not None:
            print(f"Exception occurred: {exc_type.__name__}: {exc_val}")
        return False  # Don't suppress exceptions

with FileManager("test.txt", "w") as f:
    f.write("Hello, World!")

# Creating custom context manager - function-based
from contextlib import contextmanager

@contextmanager
def file_manager(filename, mode):
    print(f"Opening {filename}")
    file = open(filename, mode)
    try:
        yield file  # Control returns to with block here
    finally:
        print(f"Closing {filename}")
        file.close()

with file_manager("test.txt", "r") as f:
    content = f.read()

# Timer context manager
@contextmanager
def timer(label):
    import time
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{label} took {end - start:.4f} seconds")

with timer("Data processing"):
    # Expensive operation
    sum(range(1000000))

# Database transaction context manager
@contextmanager
def transaction(connection):
    cursor = connection.cursor()
    try:
        yield cursor
        connection.commit()
        print("Transaction committed")
    except Exception as e:
        connection.rollback()
        print(f"Transaction rolled back: {e}")
        raise

# Usage:
# with transaction(db_connection) as cursor:
#     cursor.execute("INSERT INTO users VALUES (?)", ("John",))
#     cursor.execute("UPDATE accounts SET balance = balance - 100")

# Temporary directory context manager
import tempfile
import shutil

@contextmanager
def temporary_directory():
    temp_dir = tempfile.mkdtemp()
    try:
        yield temp_dir
    finally:
        shutil.rmtree(temp_dir)

with temporary_directory() as tmpdir:
    # Work with temporary directory
    filepath = os.path.join(tmpdir, "temp_file.txt")
    with open(filepath, "w") as f:
        f.write("Temporary data")
# Directory and contents automatically deleted

# Changing directory context manager
import os

@contextmanager
def working_directory(path):
    original_dir = os.getcwd()
    try:
        os.chdir(path)
        yield
    finally:
        os.chdir(original_dir)

with working_directory("/tmp"):
    # Work in /tmp directory
    print(os.getcwd())  # /tmp
print(os.getcwd())  # Back to original directory

# Suppressing exceptions
from contextlib import suppress

# Instead of try-except for specific exceptions
with suppress(FileNotFoundError):
    os.remove("nonexistent_file.txt")
# No error raised

with suppress(ValueError, TypeError):
    int("not a number")
# Suppresses both ValueError and TypeError

# Redirecting stdout
from contextlib import redirect_stdout
import io

f = io.StringIO()
with redirect_stdout(f):
    print("This goes to StringIO")
    print("Not to console")

output = f.getvalue()
print(output)  # Now prints to console

# Lock context manager for threading
import threading

lock = threading.Lock()

def thread_safe_operation():
    with lock:
        # Critical section - only one thread at a time
        shared_resource.modify()

# ExitStack for dynamic number of context managers
from contextlib import ExitStack

def process_files(filenames):
    with ExitStack() as stack:
        # Open all files
        files = [stack.enter_context(open(fn)) for fn in filenames]
        # Process files
        for f in files:
            process(f.read())
    # All files automatically closed

# Nested context managers
@contextmanager
def managed_resource(name):
    print(f"Acquiring {name}")
    try:
        yield name
    finally:
        print(f"Releasing {name}")

with managed_resource("Resource A"):
    with managed_resource("Resource B"):
        print("Using both resources")

# AsyncIO context managers (Python 3.7+)
import asyncio

class AsyncResource:
    async def __aenter__(self):
        print("Acquiring async resource")
        await asyncio.sleep(0.1)
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Releasing async resource")
        await asyncio.sleep(0.1)
        return False

async def use_async_resource():
    async with AsyncResource() as resource:
        print("Using async resource")

# asyncio.run(use_async_resource())

# Best practices:
# 1. Always use context managers for resources (files, connections, locks)
# 2. Prefer function-based (@contextmanager) for simple cases
# 3. Use class-based for complex cleanup logic or reusable managers
# 4. Context managers ensure cleanup even when exceptions occur

---

## Section 2: Object-Oriented Programming & Design Patterns (Q21-35)

### Q21: How do Python classes and inheritance work?

**Answer:**
Python supports single and multiple inheritance with Method Resolution Order (MRO) for handling conflicts.

```python
# Basic class
class Dog:
    # Class variable (shared by all instances)
    species = "Canis familiaris"
    
    def __init__(self, name, age):
        # Instance variables (unique to each instance)
        self.name = name
        self.age = age
    
    def bark(self):
        return f"{self.name} says Woof!"
    
    def __str__(self):
        return f"{self.name} is {self.age} years old"

# Creating instances
dog1 = Dog("Buddy", 3)
dog2 = Dog("Max", 5)

print(dog1.bark())  # "Buddy says Woof!"
print(dog1.species)  # "Canis familiaris"
print(dog1)  # "Buddy is 3 years old"

# Single inheritance
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        raise NotImplementedError("Subclass must implement speak()")

class Dog(Animal):
    def speak(self):
        return f"{self.name} says Woof!"

class Cat(Animal):
    def speak(self):
        return f"{self.name} says Meow!"

# Polymorphism
animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(animal.speak())

# Calling parent methods with super()
class Vehicle:
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
    
    def info(self):
        return f"{self.brand} {self.model}"

class Car(Vehicle):
    def __init__(self, brand, model, doors):
        super().__init__(brand, model)  # Call parent constructor
        self.doors = doors
    
    def info(self):
        parent_info = super().info()  # Call parent method
        return f"{parent_info} with {self.doors} doors"

car = Car("Toyota", "Camry", 4)
print(car.info())  # "Toyota Camry with 4 doors"

# Multiple inheritance
class Flyer:
    def fly(self):
        return "Flying in the sky"

class Swimmer:
    def swim(self):
        return "Swimming in water"

class Duck(Animal, Flyer, Swimmer):
    def speak(self):
        return f"{self.name} says Quack!"

duck = Duck("Donald")
print(duck.speak())  # "Donald says Quack!"
print(duck.fly())    # "Flying in the sky"
print(duck.swim())   # "Swimming in water"

# Method Resolution Order (MRO)
print(Duck.__mro__)
# (<class 'Duck'>, <class 'Animal'>, <class 'Flyer'>, <class 'Swimmer'>, <class 'object'>)

# Diamond problem
class A:
    def method(self):
        return "A"

class B(A):
    def method(self):
        return "B"

class C(A):
    def method(self):
        return "C"

class D(B, C):
    pass

d = D()
print(d.method())  # "B" - follows MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)

# Property decorators
class Circle:
    def __init__(self, radius):
        self._radius = radius
    
    @property
    def radius(self):
        return self._radius
    
    @radius.setter
    def radius(self, value):
        if value < 0:
            raise ValueError("Radius cannot be negative")
        self._radius = value
    
    @property
    def area(self):
        return 3.14159 * self._radius ** 2
    
    @property
    def circumference(self):
        return 2 * 3.14159 * self._radius

circle = Circle(5)
print(circle.radius)  # 5
print(circle.area)    # 78.53975
circle.radius = 10    # Uses setter
# circle.area = 100   # Error: can't set attribute

# Private and protected attributes
class BankAccount:
    def __init__(self, balance):
        self._balance = balance  # Protected (by convention)
        self.__pin = "1234"      # Name mangling (private)
    
    def get_balance(self):
        return self._balance
    
    def _internal_method(self):  # Protected method
        return "Internal use"
    
    def __verify_pin(self, pin):  # Private method
        return pin == self.__pin

account = BankAccount(1000)
print(account._balance)  # Works but discouraged
# print(account.__pin)   # AttributeError
print(account._BankAccount__pin)  # Name mangling - still accessible

# Class methods and static methods
class MathOperations:
    @staticmethod
    def add(a, b):
        return a + b
    
    @classmethod
    def multiply_by_two(cls, a):
        return cls.add(a, a)  # Can call other class methods

print(MathOperations.add(5, 3))  # 8
print(MathOperations.multiply_by_two(5))  # 10

# Abstract base classes
from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self):
        pass
    
    @abstractmethod
    def perimeter(self):
        pass

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

# shape = Shape()  # TypeError: Can't instantiate abstract class
rect = Rectangle(5, 3)
print(rect.area())  # 15

---

### Q22: What are dataclasses and when should you use them?

**Answer:**
Dataclasses (Python 3.7+) reduce boilerplate for classes that primarily store data.

```python
from dataclasses import dataclass, field
from typing import List

# Without dataclass - lots of boilerplate
class PersonOld:
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email
    
    def __repr__(self):
        return f"Person(name={self.name}, age={self.age}, email={self.email})"
    
    def __eq__(self, other):
        if not isinstance(other, PersonOld):
            return False
        return (self.name, self.age, self.email) == (other.name, other.age, other.email)

# With dataclass - clean and concise
@dataclass
class Person:
    name: str
    age: int
    email: str

# Automatically generates __init__, __repr__, __eq__
person = Person("Alice", 30, "alice@example.com")
print(person)  # Person(name='Alice', age=30, email='alice@example.com')

person2 = Person("Alice", 30, "alice@example.com")
print(person == person2)  # True - __eq__ generated automatically

# Default values
@dataclass
class Product:
    name: str
    price: float
    quantity: int = 0  # Default value
    tags: List[str] = field(default_factory=list)  # Mutable default

product = Product("Laptop", 999.99)
print(product)  # Product(name='Laptop', price=999.99, quantity=0, tags=[])

# Frozen dataclasses (immutable)
@dataclass(frozen=True)
class Point:
    x: int
    y: int

point = Point(10, 20)
# point.x = 30  # FrozenInstanceError

# Can be used as dictionary keys
points_dict = {Point(0, 0): "origin", Point(1, 1): "diagonal"}

# Order comparison
@dataclass(order=True)
class Student:
    name: str
    grade: float

students = [
    Student("Alice", 85.5),
    Student("Bob", 92.0),
    Student("Charlie", 78.5)
]
print(sorted(students))  # Sorted by all fields in order

# Custom ordering with sort_index
@dataclass(order=True)
class Task:
    priority: int = field(compare=True)
    name: str = field(compare=False)  # Don't use in comparison

task1 = Task(1, "Important")
task2 = Task(2, "Less Important")
print(task1 < task2)  # True - compared by priority only

# Post-initialization processing
@dataclass
class Rectangle:
    width: float
    height: float
    area: float = field(init=False)  # Calculated, not in __init__
    
    def __post_init__(self):
        self.area = self.width * self.height

rect = Rectangle(10, 5)
print(rect.area)  # 50

# Inheritance with dataclasses
@dataclass
class Animal:
    name: str
    age: int

@dataclass
class Dog(Animal):
    breed: str

dog = Dog("Buddy", 3, "Golden Retriever")
print(dog)  # Dog(name='Buddy', age=3, breed='Golden Retriever')

# Converting to dict and tuple
from dataclasses import asdict, astuple

person = Person("Bob", 25, "bob@example.com")
person_dict = asdict(person)
print(person_dict)  # {'name': 'Bob', 'age': 25, 'email': 'bob@example.com'}

person_tuple = astuple(person)
print(person_tuple)  # ('Bob', 25, 'bob@example.com')

# Field metadata
@dataclass
class User:
    username: str = field(metadata={"description": "Unique username"})
    password: str = field(repr=False, metadata={"sensitive": True})
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())

user = User("john_doe", "secret123")
print(user)  # password not shown in repr

# Replace method (create modified copy)
@dataclass(frozen=True)
class Config:
    host: str
    port: int
    debug: bool

config = Config("localhost", 8080, False)
new_config = config.__class__(config.host, config.port, True)
# Or use replace (if using Python 3.10+)
from dataclasses import replace
# new_config = replace(config, debug=True)

# Complex example: nested dataclasses
@dataclass
class Address:
    street: str
    city: str
    zipcode: str

@dataclass
class Employee:
    name: str
    age: int
    address: Address
    skills: List[str] = field(default_factory=list)
    
    def add_skill(self, skill: str):
        self.skills.append(skill)

address = Address("123 Main St", "Boston", "02101")
employee = Employee("Alice", 30, address)
employee.add_skill("Python")
employee.add_skill("JavaScript")
print(employee)

# When to use dataclasses:
# ✓ Data containers (DTOs, config objects, API models)
# ✓ When you need __repr__, __eq__ automatically
# ✓ Immutable data structures (frozen=True)
# ✓ Simple value objects
# ✗ Complex business logic (use regular classes)
# ✗ When you need custom __init__ logic

---

### Q23: What are Python's magic/dunder methods and when do you use them?

**Answer:**
Magic methods (double underscore methods) let you customize how your classes behave with Python's built-in operations.

```python
# Complete example with most useful magic methods
class Vector:
    def __init__(self, x, y):
        """Constructor"""
        self.x = x
        self.y = y
    
    # String representations
    def __repr__(self):
        """Official string representation - for developers"""
        return f"Vector({self.x}, {self.y})"
    
    def __str__(self):
        """Informal string representation - for users"""
        return f"({self.x}, {self.y})"
    
    # Arithmetic operators
    def __add__(self, other):
        """v1 + v2"""
        return Vector(self.x + other.x, self.y + other.y)
    
    def __sub__(self, other):
        """v1 - v2"""
        return Vector(self.x - other.x, self.y - other.y)
    
    def __mul__(self, scalar):
        """v * 5"""
        return Vector(self.x * scalar, self.y * scalar)
    
    def __rmul__(self, scalar):
        """5 * v (reverse multiplication)"""
        return self * scalar
    
    def __truediv__(self, scalar):
        """v / 2"""
        return Vector(self.x / scalar, self.y / scalar)
    
    # Comparison operators
    def __eq__(self, other):
        """v1 == v2"""
        return self.x == other.x and self.y == other.y
    
    def __ne__(self, other):
        """v1 != v2"""
        return not self == other
    
    def __lt__(self, other):
        """v1 < v2 (by magnitude)"""
        return self.magnitude() < other.magnitude()
    
    # Container operations
    def __len__(self):
        """len(v) - returns 2 for 2D vector"""
        return 2
    
    def __getitem__(self, index):
        """v[0] or v[1]"""
        if index == 0:
            return self.x
        elif index == 1:
            return self.y
        raise IndexError("Vector index out of range")
    
    def __setitem__(self, index, value):
        """v[0] = 10"""
        if index == 0:
            self.x = value
        elif index == 1:
            self.y = value
        else:
            raise IndexError("Vector index out of range")
    
    # Context manager
    def __enter__(self):
        print(f"Using vector {self}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        print(f"Done using vector")
        return False
    
    # Callable
    def __call__(self, scalar):
        """v(5) - makes instance callable"""
        return self * scalar
    
    # Boolean context
    def __bool__(self):
        """bool(v) - False if zero vector"""
        return self.x != 0 or self.y != 0
    
    # Hash (for use in sets and as dict keys)
    def __hash__(self):
        """hash(v) - allows use in sets/dict keys"""
        return hash((self.x, self.y))
    
    # Utility
    def magnitude(self):
        return (self.x**2 + self.y**2)**0.5

# Using the Vector class
v1 = Vector(3, 4)
v2 = Vector(1, 2)

print(v1)  # (3, 4) - uses __str__
print(repr(v1))  # Vector(3, 4) - uses __repr__

v3 = v1 + v2  # Uses __add__
print(v3)  # (4, 6)

v4 = v1 * 2  # Uses __mul__
v5 = 2 * v1  # Uses __rmul__
print(v4, v5)  # Both (6, 8)

print(v1 == Vector(3, 4))  # True - uses __eq__
print(v1 < v2)  # False - uses __lt__

print(len(v1))  # 2 - uses __len__
print(v1[0], v1[1])  # 3 4 - uses __getitem__
v1[0] = 10  # Uses __setitem__

with v1:  # Uses __enter__ and __exit__
    print("Working with vector")

result = v1(3)  # Uses __call__
print(result)  # (30, 12)

if v1:  # Uses __bool__
    print("Non-zero vector")

# Can be used in sets
vectors = {Vector(1, 2), Vector(3, 4), Vector(1, 2)}
print(len(vectors))  # 2 - duplicates removed

# More useful magic methods

# __format__ - custom string formatting
class Money:
    def __init__(self, amount, currency="USD"):
        self.amount = amount
        self.currency = currency
    
    def __format__(self, format_spec):
        if format_spec == 'c':
            symbols = {"USD": "$", "EUR": "€", "GBP": "£"}
            symbol = symbols.get(self.currency, self.currency)
            return f"{symbol}{self.amount:.2f}"
        return str(self.amount)

price = Money(99.99, "USD")
print(f"Price: {price:c}")  # "Price: $99.99"

# __iter__ and __next__ - make class iterable
class Countdown:
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for num in Countdown(5):
    print(num)  # 5, 4, 3, 2, 1

# __contains__ - for 'in' operator
class Inventory:
    def __init__(self):
        self.items = {}
    
    def add(self, item, quantity):
        self.items[item] = quantity
    
    def __contains__(self, item):
        return item in self.items

inventory = Inventory()
inventory.add("apple", 10)
print("apple" in inventory)  # True

# __del__ - destructor (use with caution)
class Resource:
    def __init__(self, name):
        self.name = name
        print(f"Acquiring {name}")
    
    def __del__(self):
        print(f"Releasing {self.name}")
# Note: __del__ is not guaranteed to be called!
# Use context managers instead for cleanup

# __getattr__ and __setattr__ - attribute access
class DynamicAttributes:
    def __getattr__(self, name):
        """Called when attribute not found"""
        return f"'{name}' not found"
    
    def __setattr__(self, name, value):
        """Called on every attribute assignment"""
        print(f"Setting {name} = {value}")
        super().__setattr__(name, value)

obj = DynamicAttributes()
print(obj.nonexistent)  # "'nonexistent' not found"
obj.x = 10  # "Setting x = 10"

---

### Q24: What is method resolution order (MRO) and why does it matter?

**Answer:**
MRO determines the order Python searches for methods in inheritance hierarchies, especially with multiple inheritance.

```python
# Simple inheritance - easy to understand
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")

b = B()
b.method()  # "B" - straightforward

# Multiple inheritance - more complex
class A:
    def method(self):
        print("A")

class B(A):
    def method(self):
        print("B")

class C(A):
    def method(self):
        print("C")

class D(B, C):
    pass

d = D()
d.method()  # "B" - but why?

# Check MRO
print(D.__mro__)
# (<class 'D'>, <class 'B'>, <class 'C'>, <class 'A'>, <class 'object'>)
# Searches in this order: D -> B -> C -> A -> object

# C3 Linearization algorithm
# Rules:
# 1. Child classes come before parents
# 2. Parent order is preserved as specified
# 3. For each class, its parents appear in the same order

# Diamond problem
class Base:
    def __init__(self):
        print("Base.__init__")

class A(Base):
    def __init__(self):
        print("A.__init__")
        super().__init__()

class B(Base):
    def __init__(self):
        print("B.__init__")
        super().__init__()

class C(A, B):
    def __init__(self):
        print("C.__init__")
        super().__init__()

c = C()
# C.__init__
# A.__init__
# B.__init__
# Base.__init__
# Each __init__ called exactly once!

print(C.__mro__)
# (<class 'C'>, <class 'A'>, <class 'B'>, <class 'Base'>, <class 'object'>)

# Why super() is important
class Rectangle:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class ColoredRectangle(Rectangle):
    def __init__(self, width, height, color):
        super().__init__(width, height)  # Call parent init
        self.color = color

# Without super() - problems in diamond inheritance
class A:
    def __init__(self):
        self.value_a = "A"

class B(A):
    def __init__(self):
        A.__init__(self)  # Direct call - problematic!
        self.value_b = "B"

class C(A):
    def __init__(self):
        A.__init__(self)  # Direct call - problematic!
        self.value_c = "C"

class D(B, C):
    def __init__(self):
        B.__init__(self)  # A.__init__ called
        C.__init__(self)  # A.__init__ called AGAIN!
# Base class initialized twice!

# With super() - correct
class A:
    def __init__(self):
        print("A init")
        self.value_a = "A"

class B(A):
    def __init__(self):
        super().__init__()  # Follows MRO
        print("B init")
        self.value_b = "B"

class C(A):
    def __init__(self):
        super().__init__()  # Follows MRO
        print("C init")
        self.value_c = "C"

class D(B, C):
    def __init__(self):
        super().__init__()  # Follows MRO
        print("D init")

d = D()
# A init (once!)
# C init
# B init
# D init

# Practical example: Mixins
class JSONMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)

class TimestampMixin:
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from datetime import datetime
        self.created_at = datetime.now()

class User(TimestampMixin, JSONMixin):
    def __init__(self, username, email):
        super().__init__()
        self.username = username
        self.email = email

user = User("john_doe", "john@example.com")
print(user.to_json())  # Includes created_at timestamp

# Checking MRO
print(User.__mro__)

# MRO conflicts - when Python can't determine order
# This will raise TypeError:
# class A: pass
# class B(A): pass
# class C(A): pass
# class D(B, A):  # Error! A already appears before B in MRO
#     pass

# Best practices with MRO:
# 1. Use super() instead of direct parent calls
# 2. Design inheritance hierarchies carefully
# 3. Keep diamond patterns simple
# 4. Use mixins for cross-cutting concerns
# 5. Check MRO with __mro__ when debugging

---

### Q25: What are design patterns in Python and how do you implement them?

**Answer:**
Design patterns are reusable solutions to common problems. Here are the most important ones in Python.

```python
# 1. SINGLETON PATTERN - Single instance of a class
class Singleton:
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

s1 = Singleton()
s2 = Singleton()
print(s1 is s2)  # True

# Better singleton with decorator
def singleton(cls):
    instances = {}
    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return get_instance

@singleton
class DatabaseConnection:
    def __init__(self):
        print("Creating connection")

db1 = DatabaseConnection()  # "Creating connection"
db2 = DatabaseConnection()  # Nothing printed
print(db1 is db2)  # True

# 2. FACTORY PATTERN - Create objects without specifying exact class
class Animal:
    def speak(self):
        pass

class Dog(Animal):
    def speak(self):
        return "Woof!"

class Cat(Animal):
    def speak(self):
        return "Meow!"

class AnimalFactory:
    @staticmethod
    def create_animal(animal_type):
        if animal_type == "dog":
            return Dog()
        elif animal_type == "cat":
            return Cat()
        raise ValueError(f"Unknown animal type: {animal_type}")

# Usage
animal = AnimalFactory.create_animal("dog")
print(animal.speak())  # "Woof!"

# 3. BUILDER PATTERN - Construct complex objects step by step
class Pizza:
    def __init__(self):
        self.dough = None
        self.sauce = None
        self.toppings = []
    
    def __str__(self):
        return f"Pizza with {self.dough} dough, {self.sauce} sauce, toppings: {', '.join(self.toppings)}"

class PizzaBuilder:
    def __init__(self):
        self.pizza = Pizza()
    
    def set_dough(self, dough):
        self.pizza.dough = dough
        return self  # Method chaining
    
    def set_sauce(self, sauce):
        self.pizza.sauce = sauce
        return self
    
    def add_topping(self, topping):
        self.pizza.toppings.append(topping)
        return self
    
    def build(self):
        return self.pizza

# Usage with method chaining
pizza = (PizzaBuilder()
         .set_dough("thin crust")
         .set_sauce("tomato")
         .add_topping("cheese")
         .add_topping("pepperoni")
         .build())
print(pizza)

# 4. OBSERVER PATTERN - Subscribe to and receive notifications
class Subject:
    def __init__(self):
        self._observers = []
        self._state = None
    
    def attach(self, observer):
        self._observers.append(observer)
    
    def detach(self, observer):
        self._observers.remove(observer)
    
    def notify(self):
        for observer in self._observers:
            observer.update(self._state)
    
    def set_state(self, state):
        self._state = state
        self.notify()

class Observer:
    def __init__(self, name):
        self.name = name
    
    def update(self, state):
        print(f"{self.name} received update: {state}")

# Usage
subject = Subject()
obs1 = Observer("Observer 1")
obs2 = Observer("Observer 2")

subject.attach(obs1)
subject.attach(obs2)

subject.set_state("New State")
# Observer 1 received update: New State
# Observer 2 received update: New State

# 5. STRATEGY PATTERN - Different algorithms, same interface
class PaymentStrategy:
    def pay(self, amount):
        pass

class CreditCardPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} with credit card"

class PayPalPayment(PaymentStrategy):
    def pay(self, amount):
        return f"Paid ${amount} with PayPal"

class ShoppingCart:
    def __init__(self, payment_strategy):
        self.payment_strategy = payment_strategy
    
    def checkout(self, amount):
        return self.payment_strategy.pay(amount)

# Usage
cart = ShoppingCart(CreditCardPayment())
print(cart.checkout(100))

cart.payment_strategy = PayPalPayment()
print(cart.checkout(50))

# 6. DECORATOR PATTERN - Add functionality dynamically
class Coffee:
    def cost(self):
        return 5
    
    def description(self):
        return "Coffee"

class MilkDecorator:
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost() + 1
    
    def description(self):
        return self._coffee.description() + ", Milk"

class SugarDecorator:
    def __init__(self, coffee):
        self._coffee = coffee
    
    def cost(self):
        return self._coffee.cost() + 0.5
    
    def description(self):
        return self._coffee.description() + ", Sugar"

# Usage
coffee = Coffee()
coffee = MilkDecorator(coffee)
coffee = SugarDecorator(coffee)
print(f"{coffee.description()}: ${coffee.cost()}")
# "Coffee, Milk, Sugar: $6.5"

# 7. ADAPTER PATTERN - Make incompatible interfaces work together
class EuropeanSocket:
    def provide_electricity(self):
        return "230V"

class USDevice:
    def connect(self, voltage):
        if voltage == "110V":
            return "Device powered"
        return "Voltage mismatch"

class VoltageAdapter:
    def __init__(self, socket):
        self.socket = socket
    
    def provide_power(self):
        eu_voltage = self.socket.provide_electricity()
        # Convert 230V to 110V
        return "110V"

# Usage
socket = EuropeanSocket()
adapter = VoltageAdapter(socket)
device = USDevice()
print(device.connect(adapter.provide_power()))  # "Device powered"

# 8. DEPENDENCY INJECTION - Provide dependencies from outside
# Bad: Hard-coded dependency
class UserServiceBad:
    def __init__(self):
        self.db = MySQLDatabase()  # Tightly coupled!

# Good: Inject dependency
class UserService:
    def __init__(self, database):
        self.db = database  # Can be any database
    
    def get_user(self, user_id):
        return self.db.fetch(user_id)

class MySQLDatabase:
    def fetch(self, id):
        return f"User from MySQL: {id}"

class PostgreSQLDatabase:
    def fetch(self, id):
        return f"User from PostgreSQL: {id}"

# Usage - easy to swap implementations
service = UserService(MySQLDatabase())
print(service.get_user(1))

service = UserService(PostgreSQLDatabase())
print(service.get_user(1))

---

### Q26: How do you implement properties and descriptors in Python?

**Answer:**
Properties and descriptors control attribute access, enabling validation, computed attributes, and more.

```python
# Basic property
class Temperature:
    def __init__(self, celsius):
        self._celsius = celsius
    
    @property
    def celsius(self):
        """Get temperature in Celsius"""
        return self._celsius
    
    @celsius.setter
    def celsius(self, value):
        """Set temperature in Celsius"""
        if value < -273.15:
            raise ValueError("Temperature below absolute zero")
        self._celsius = value
    
    @property
    def fahrenheit(self):
        """Get temperature in Fahrenheit"""
        return self._celsius * 9/5 + 32
    
    @fahrenheit.setter
    def fahrenheit(self, value):
        """Set temperature via Fahrenheit"""
        self.celsius = (value - 32) * 5/9

temp = Temperature(25)
print(temp.celsius)  # 25
print(temp.fahrenheit)  # 77.0

temp.fahrenheit = 32
print(temp.celsius)  # 0.0

# Read-only property (no setter)
class Circle:
    def __init__(self, radius):
        self.radius = radius
    
    @property
    def area(self):
        return 3.14159 * self.radius ** 2
    
    @property
    def circumference(self):
        return 2 * 3.14159 * self.radius

circle = Circle(5)
print(circle.area)  # 78.53975
# circle.area = 100  # AttributeError: can't set attribute

# Lazy property - computed once
class DataSet:
    def __init__(self, data):
        self._data = data
        self._mean = None
    
    @property
    def mean(self):
        if self._mean is None:
            print("Computing mean...")
            self._mean = sum(self._data) / len(self._data)
        return self._mean

dataset = DataSet([1, 2, 3, 4, 5])
print(dataset.mean)  # "Computing mean..." then 3.0
print(dataset.mean)  # 3.0 (no recomputation)

# Descriptor protocol - for reusable validation
class Validator:
    def __init__(self, min_value=None, max_value=None):
        self.min_value = min_value
        self.max_value = max_value
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if self.min_value is not None and value < self.min_value:
            raise ValueError(f"{self.name} must be >= {self.min_value}")
        if self.max_value is not None and value > self.max_value:
            raise ValueError(f"{self.name} must be <= {self.max_value}")
        instance.__dict__[self.name] = value

class Person:
    age = Validator(min_value=0, max_value=150)
    height = Validator(min_value=0)
    
    def __init__(self, age, height):
        self.age = age
        self.height = height

person = Person(30, 180)
print(person.age)  # 30
# person.age = -5  # ValueError
# person.age = 200  # ValueError

# Type checking descriptor
class TypedProperty:
    def __init__(self, expected_type):
        self.expected_type = expected_type
    
    def __set_name__(self, owner, name):
        self.name = name
    
    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.get(self.name)
    
    def __set__(self, instance, value):
        if not isinstance(value, self.expected_type):
            raise TypeError(
                f"{self.name} must be {self.expected_type.__name__}"
            )
        instance.__dict__[self.name] = value

class Product:
    name = TypedProperty(str)
    price = TypedProperty(float)
    quantity = TypedProperty(int)
    
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity

product = Product("Laptop", 999.99, 5)
# product.price = "expensive"  # TypeError

# Cached property (Python 3.8+)
from functools import cached_property

class WebPage:
    def __init__(self, url):
        self.url = url
    
    @cached_property
    def content(self):
        print(f"Fetching {self.url}")
        # Expensive operation
        return f"Content from {self.url}"

page = WebPage("https://example.com")
print(page.content)  # "Fetching..." then content
print(page.content)  # Just content (cached)

---

### Q27: What are metaclasses and when should you use them?

**Answer:**
Metaclasses are "classes of classes" - they control class creation. Use them rarely, for frameworks and advanced magic.

```python
# Basic metaclass
class Meta(type):
    def __new__(mcs, name, bases, namespace):
        print(f"Creating class {name}")
        return super().__new__(mcs, name, bases, namespace)

class MyClass(metaclass=Meta):
    pass  # "Creating class MyClass"

# Singleton via metaclass
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    def __init__(self):
        print("Creating database connection")

db1 = Database()  # "Creating database connection"
db2 = Database()  # Nothing printed
print(db1 is db2)  # True

# Auto-register subclasses
class PluginMeta(type):
    plugins = []
    
    def __new__(mcs, name, bases, namespace):
        cls = super().__new__(mcs, name, bases, namespace)
        if name != 'Plugin':  # Don't register base class
            mcs.plugins.append(cls)
        return cls

class Plugin(metaclass=PluginMeta):
    pass

class EmailPlugin(Plugin):
    pass

class SMSPlugin(Plugin):
    pass

print(PluginMeta.plugins)  # [EmailPlugin, SMSPlugin]

# When NOT to use metaclasses:
# - Decorators can solve most problems
# - Class decorators are simpler
# - Only use for framework-level magic

# Most people never need metaclasses!
# "Metaclasses are deeper magic than 99% of users should ever worry about" 
# - Tim Peters

---

### Q28: How do you properly handle class and instance attributes?

**Answer:**
Understanding the difference between class and instance attributes is crucial for proper object-oriented design.

```python
class MyClass:
    class_var = []  # Class attribute - shared by all instances
    
    def __init__(self):
        self.instance_var = []  # Instance attribute - unique to each instance

obj1 = MyClass()
obj2 = MyClass()

# Instance attributes are independent
obj1.instance_var.append(1)
print(obj1.instance_var)  # [1]
print(obj2.instance_var)  # []

# Class attributes are shared!
obj1.class_var.append(1)
print(obj1.class_var)  # [1]
print(obj2.class_var)  # [1] - same list!
print(MyClass.class_var)  # [1]

# Proper use of class attributes
class Counter:
    count = 0  # Class attribute for counting instances
    
    def __init__(self, name):
        self.name = name
        Counter.count += 1  # Modify class attribute
    
    @classmethod
    def get_count(cls):
        return cls.count

c1 = Counter("first")
c2 = Counter("second")
print(Counter.get_count())  # 2

# Mutable default argument problem - WRONG
class WrongClass:
    def __init__(self, items=[]):  # BAD!
        self.items = items

obj1 = WrongClass()
obj2 = WrongClass()
obj1.items.append(1)
print(obj2.items)  # [1] - Surprise!

# Correct approach
class CorrectClass:
    def __init__(self, items=None):
        self.items = items if items is not None else []

obj1 = CorrectClass()
obj2 = CorrectClass()
obj1.items.append(1)
print(obj2.items)  # []

# Using class attributes for constants
class Config:
    MAX_CONNECTIONS = 100
    TIMEOUT = 30
    DEFAULT_PORT = 8080

# Name mangling for "private" attributes
class BankAccount:
    def __init__(self, balance):
        self.__balance = balance  # Name mangled to _BankAccount__balance
    
    def get_balance(self):
        return self.__balance

account = BankAccount(1000)
# print(account.__balance)  # AttributeError
print(account._BankAccount__balance)  # 1000 - can still access

---

### Q29: What are slots and when should you use them?

**Answer:**
Slots optimize memory by declaring a fixed set of attributes, preventing the creation of `__dict__`.

```python
# Regular class - uses __dict__
class RegularPerson:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Class with slots - more memory efficient
class SlottedPerson:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Compare memory usage
import sys

regular = RegularPerson("Alice", 30)
slotted = SlottedPerson("Bob", 25)

print(sys.getsizeof(regular.__dict__))  # ~120 bytes
# print(sys.getsizeof(slotted.__dict__))  # AttributeError - no __dict__!

# With slots, you can't add new attributes dynamically
# slotted.email = "bob@example.com"  # AttributeError

# Regular class allows dynamic attributes
regular.email = "alice@example.com"  # Works fine

# Slots with inheritance
class Person:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Employee(Person):
    __slots__ = ['employee_id']  # Additional slots
    
    def __init__(self, name, age, employee_id):
        super().__init__(name, age)
        self.employee_id = employee_id

emp = Employee("Charlie", 35, "E123")

# Memory savings example - creating millions of objects
import time

def measure_memory_and_time(cls, n=1_000_000):
    start_time = time.time()
    objects = [cls(f"Person{i}", i % 100) for i in range(n)]
    end_time = time.time()
    
    # Approximate memory (simplified)
    memory = sys.getsizeof(objects[0]) * n / (1024 * 1024)  # MB
    
    print(f"{cls.__name__}:")
    print(f"  Time: {end_time - start_time:.2f}s")
    print(f"  Approx memory per object: {sys.getsizeof(objects[0])} bytes")

# When to use slots:
# ✓ Creating many instances (millions)
# ✓ Performance-critical code
# ✓ When you know exact attributes needed
# ✗ When you need dynamic attributes
# ✗ When using multiple inheritance (complex)
# ✗ When flexibility is more important than performance

---

### Q30: How do abstract base classes (ABC) work?

**Answer:**
ABCs define interfaces that subclasses must implement, enforcing a contract.

```python
from abc import ABC, abstractmethod

# Define abstract base class
class Shape(ABC):
    @abstractmethod
    def area(self):
        """Calculate area - must be implemented by subclasses"""
        pass
    
    @abstractmethod
    def perimeter(self):
        """Calculate perimeter - must be implemented by subclasses"""
        pass
    
    def describe(self):
        """Concrete method - can be used as-is"""
        return f"Shape with area {self.area()}"

# Cannot instantiate abstract class
# shape = Shape()  # TypeError: Can't instantiate abstract class

# Concrete implementation
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height
    
    def perimeter(self):
        return 2 * (self.width + self.height)

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2
    
    def perimeter(self):
        return 2 * 3.14159 * self.radius

# Now we can create instances
rect = Rectangle(5, 3)
print(rect.area())  # 15
print(rect.describe())  # "Shape with area 15"

circle = Circle(5)
print(circle.area())  # 78.53975

# Incomplete implementation fails
class IncompleteShape(Shape):
    def area(self):
        return 0
    # Missing perimeter() implementation

# incomplete = IncompleteShape()  # TypeError

# Abstract properties
class Vehicle(ABC):
    @property
    @abstractmethod
    def wheels(self):
        pass
    
    @abstractmethod
    def drive(self):
        pass

class Car(Vehicle):
    @property
    def wheels(self):
        return 4
    
    def drive(self):
        return "Driving on road"

car = Car()
print(car.wheels)  # 4

# Abstract class methods
class Database(ABC):
    @classmethod
    @abstractmethod
    def connect(cls, connection_string):
        pass
    
    @staticmethod
    @abstractmethod
    def validate_query(query):
        pass

class MySQL(Database):
    @classmethod
    def connect(cls, connection_string):
        return f"Connected to MySQL: {connection_string}"
    
    @staticmethod
    def validate_query(query):
        return "SELECT" in query or "INSERT" in query

# Using ABCs for duck typing validation
from collections.abc import Sized, Iterable

class CustomCollection(Sized, Iterable):
    def __init__(self, items):
        self._items = items
    
    def __len__(self):
        return len(self._items)
    
    def __iter__(self):
        return iter(self._items)

collection = CustomCollection([1, 2, 3])
print(len(collection))  # 3
print(list(collection))  # [1, 2, 3]

# Check if object implements an ABC
print(isinstance(collection, Sized))  # True
print(isinstance(collection, Iterable))  # True

---

### Q31: How do you implement operator overloading effectively?

**Answer:**
Operator overloading lets custom objects work with Python's built-in operators naturally.

```python
class Money:
    def __init__(self, amount, currency="USD"):
        self.amount = amount
        self.currency = currency
    
    def __repr__(self):
        return f"Money({self.amount}, {self.currency})"
    
    def __str__(self):
        return f"${self.amount} {self.currency}"
    
    # Addition
    def __add__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError("Cannot add different currencies")
            return Money(self.amount + other.amount, self.currency)
        elif isinstance(other, (int, float)):
            return Money(self.amount + other, self.currency)
        return NotImplemented
    
    # Right-hand addition (5 + money)
    def __radd__(self, other):
        return self.__add__(other)
    
    # Subtraction
    def __sub__(self, other):
        if isinstance(other, Money):
            if self.currency != other.currency:
                raise ValueError("Cannot subtract different currencies")
            return Money(self.amount - other.amount, self.currency)
        elif isinstance(other, (int, float)):
            return Money(self.amount - other, self.currency)
        return NotImplemented
    
    # Multiplication
    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return Money(self.amount * other, self.currency)
        return NotImplemented
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    # Division
    def __truediv__(self, other):
        if isinstance(other, (int, float)):
            return Money(self.amount / other, self.currency)
        return NotImplemented
    
    # Comparison operators
    def __eq__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        return self.amount == other.amount and self.currency == other.currency
    
    def __lt__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        if self.currency != other.currency:
            raise ValueError("Cannot compare different currencies")
        return self.amount < other.amount
    
    def __le__(self, other):
        return self == other or self < other
    
    def __gt__(self, other):
        if not isinstance(other, Money):
            return NotImplemented
        if self.currency != other.currency:
            raise ValueError("Cannot compare different currencies")
        return self.amount > other.amount
    
    def __ge__(self, other):
        return self == other or self > other

# Usage
m1 = Money(100)
m2 = Money(50)

print(m1 + m2)  # $150 USD
print(m1 - m2)  # $50 USD
print(m1 * 2)   # $200 USD
print(2 * m1)   # $200 USD (uses __rmul__)
print(m1 / 2)   # $50.0 USD

print(m1 > m2)  # True
print(m1 == Money(100))  # True

# Matrix class with operator overloading
class Matrix:
    def __init__(self, data):
        self.data = data
        self.rows = len(data)
        self.cols = len(data[0]) if data else 0
    
    def __repr__(self):
        return f"Matrix({self.data})"
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError("Matrix dimensions must match")
        
        result = [[self.data[i][j] + other.data[i][j] 
                   for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)
    
    def __mul__(self, scalar):
        result = [[self.data[i][j] * scalar 
                   for j in range(self.cols)]
                  for i in range(self.rows)]
        return Matrix(result)

m1 = Matrix([[1, 2], [3, 4]])
m2 = Matrix([[5, 6], [7, 8]])

m3 = m1 + m2
print(m3)  # Matrix([[6, 8], [10, 12]])

m4 = m1 * 2
print(m4)  # Matrix([[2, 4], [6, 8]])

---

### Q32: What are mixins and how do you use them?

**Answer:**
Mixins are classes that provide specific functionality to be mixed into other classes via multiple inheritance.

```python
# Basic mixin example
class JSONMixin:
    def to_json(self):
        import json
        return json.dumps(self.__dict__)
    
    @classmethod
    def from_json(cls, json_string):
        import json
        data = json.loads(json_string)
        return cls(**data)

class TimestampMixin:
    def set_timestamp(self):
        from datetime import datetime
        self.created_at = datetime.now().isoformat()

# Using mixins
class User(JSONMixin, TimestampMixin):
    def __init__(self, username, email):
        self.username = username
        self.email = email
        self.set_timestamp()

user = User("john_doe", "john@example.com")
json_str = user.to_json()
print(json_str)

# Logging mixin
class LoggingMixin:
    def log(self, message):
        print(f"[{self.__class__.__name__}] {message}")

class Service(LoggingMixin):
    def process(self):
        self.log("Processing started")
        # Do work
        self.log("Processing completed")

service = Service()
service.process()

# Validation mixin
class ValidationMixin:
    def validate(self):
        for attr_name, attr_type in self.__annotations__.items():
            value = getattr(self, attr_name, None)
            if value is None:
                raise ValueError(f"{attr_name} is required")
            if not isinstance(value, attr_type):
                raise TypeError(f"{attr_name} must be {attr_type.__name__}")

class Product(ValidationMixin):
    name: str
    price: float
    quantity: int
    
    def __init__(self, name, price, quantity):
        self.name = name
        self.price = price
        self.quantity = quantity
        self.validate()

product = Product("Laptop", 999.99, 5)
# product = Product("Laptop", "expensive", 5)  # TypeError

# Comparison mixin
class ComparableMixin:
    def __eq__(self, other):
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self.__dict__ == other.__dict__
    
    def __ne__(self, other):
        return not self.__eq__(other)

class Person(ComparableMixin):
    def __init__(self, name, age):
        self.name = name
        self.age = age

p1 = Person("Alice", 30)
p2 = Person("Alice", 30)
p3 = Person("Bob", 25)

print(p1 == p2)  # True
print(p1 == p3)  # False

---

### Q33: How do you implement the Iterator and Iterable protocols?

**Answer:**
Implement `__iter__()` and `__next__()` to make objects iterable and work with for loops.

```python
# Basic iterator
class Countdown:
    def __init__(self, start):
        self.start = start
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

for num in Countdown(5):
    print(num)  # 5, 4, 3, 2, 1

# Separate iterator and iterable
class NumberRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __iter__(self):
        return NumberRangeIterator(self.start, self.end)

class NumberRangeIterator:
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# Can be iterated multiple times
num_range = NumberRange(1, 5)
print(list(num_range))  # [1, 2, 3, 4]
print(list(num_range))  # [1, 2, 3, 4] - works again

# File reader iterator
class FileReader:
    def __init__(self, filename):
        self.filename = filename
        self.file = None
    
    def __iter__(self):
        self.file = open(self.filename, 'r')
        return self
    
    def __next__(self):
        line = self.file.readline()
        if not line:
            self.file.close()
            raise StopIteration
        return line.strip()

# Usage:
# for line in FileReader('data.txt'):
#     print(line)

# Infinite iterator
class InfiniteCounter:
    def __init__(self, start=0):
        self.current = start
    
    def __iter__(self):
        return self
    
    def __next__(self):
        value = self.current
        self.current += 1
        return value

counter = InfiniteCounter()
# Use with itertools.islice to limit
from itertools import islice
print(list(islice(counter, 5)))  # [0, 1, 2, 3, 4]

---

### Q34: What are protocols and structural subtyping in Python?

**Answer:**
Protocols (Python 3.8+) enable structural typing - objects are compatible if they have the right methods.

```python
from typing import Protocol, runtime_checkable

# Define a protocol
@runtime_checkable
class Drawable(Protocol):
    def draw(self) -> str:
        ...

# Classes don't need to explicitly inherit from Protocol
class Circle:
    def draw(self) -> str:
        return "Drawing a circle"

class Square:
    def draw(self) -> str:
        return "Drawing a square"

# Both work with Drawable protocol
def render(shape: Drawable) -> None:
    print(shape.draw())

circle = Circle()
square = Square()

render(circle)  # Works!
render(square)  # Works!

# Runtime checking
print(isinstance(circle, Drawable))  # True
print(isinstance(square, Drawable))  # True

# Protocol with properties
class Sized(Protocol):
    @property
    def size(self) -> int:
        ...

class Container:
    def __init__(self, items):
        self._items = items
    
    @property
    def size(self) -> int:
        return len(self._items)

container = Container([1, 2, 3])
print(isinstance(container, Sized))  # True

# Multiple method protocol
class Repository(Protocol):
    def save(self, item: object) -> None:
        ...
    
    def find(self, id: int) -> object:
        ...
    
    def delete(self, id: int) -> None:
        ...

# Any class implementing these methods is a Repository
class UserRepository:
    def __init__(self):
        self.users = {}
    
    def save(self, user):
        self.users[user.id] = user
    
    def find(self, id):
        return self.users.get(id)
    
    def delete(self, id):
        del self.users[id]

repo = UserRepository()
print(isinstance(repo, Repository))  # True

---

### Q35: How do you implement polymorphism in Python?

**Answer:**
Python supports polymorphism through duck typing, inheritance, and abstract base classes.

```python
# Duck typing polymorphism
def make_sound(animal):
    # Works with any object that has a speak() method
    return animal.speak()

class Dog:
    def speak(self):
        return "Woof!"

class Cat:
    def speak(self):
        return "Meow!"

class Car:
    def speak(self):
        return "Beep beep!"  # Anything with speak() works

print(make_sound(Dog()))  # "Woof!"
print(make_sound(Cat()))  # "Meow!"
print(make_sound(Car()))  # "Beep beep!"

# Inheritance-based polymorphism
class Shape:
    def area(self):
        raise NotImplementedError

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    
    def area(self):
        return 3.14159 * self.radius ** 2

def print_area(shape: Shape):
    print(f"Area: {shape.area()}")

shapes = [Rectangle(5, 3), Circle(4)]
for shape in shapes:
    print_area(shape)

# Operator polymorphism
def add_items(a, b):
    return a + b

print(add_items(5, 3))  # 8 (int addition)
print(add_items("Hello", " World"))  # "Hello World" (string concatenation)
print(add_items([1, 2], [3, 4]))  # [1, 2, 3, 4] (list concatenation)

# Method overriding
class Animal:
    def __init__(self, name):
        self.name = name
    
    def speak(self):
        return "Some sound"
    
    def introduce(self):
        return f"{self.name} says: {self.speak()}"

class Dog(Animal):
    def speak(self):  # Override parent method
        return "Woof!"

class Cat(Animal):
    def speak(self):  # Override parent method
        return "Meow!"

animals = [Dog("Buddy"), Cat("Whiskers")]
for animal in animals:
    print(animal.introduce())
# Buddy says: Woof!
# Whiskers says: Meow!

---

## Section 3: Functional Programming & Advanced Functions (Q36-50)

### Q36: What is functional programming and how is it applied in Python?

**Answer:**
Functional programming treats computation as the evaluation of mathematical functions, avoiding state and mutable data.

```python
# Pure functions - same input always gives same output, no side effects
def pure_add(a, b):
    return a + b  # No external state modified

# Impure function - has side effects
total = 0
def impure_add(a, b):
    global total
    total += a + b  # Modifies external state
    return total

# First-class functions - functions are objects
def square(x):
    return x ** 2

# Assign to variable
my_func = square
print(my_func(5))  # 25

# Pass as argument
def apply_function(func, value):
    return func(value)

print(apply_function(square, 5))  # 25

# Return from function
def get_operation(op):
    if op == "square":
        return lambda x: x ** 2
    elif op == "double":
        return lambda x: x * 2

operation = get_operation("square")
print(operation(5))  # 25

# Higher-order functions - map, filter, reduce
numbers = [1, 2, 3, 4, 5]

# map - transform each element
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# filter - keep elements that pass test
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]

# reduce - combine elements
from functools import reduce
sum_all = reduce(lambda x, y: x + y, numbers)
print(sum_all)  # 15

# Immutability - prefer creating new data over modifying
original = [1, 2, 3]
# Don't do: original.append(4)
# Do this:
new_list = original + [4]
print(original)  # [1, 2, 3] - unchanged
print(new_list)  # [1, 2, 3, 4]

# Function composition
def add_one(x):
    return x + 1

def double(x):
    return x * 2

def compose(f, g):
    return lambda x: f(g(x))

add_then_double = compose(double, add_one)
print(add_then_double(5))  # 12 (5+1=6, 6*2=12)

# Partial application
from functools import partial

def power(base, exponent):
    return base ** exponent

square_func = partial(power, exponent=2)
cube_func = partial(power, exponent=3)

print(square_func(5))  # 25
print(cube_func(5))   # 125

# Currying - transform function with multiple args to nested functions
def curry_add(a):
    def inner(b):
        return a + b
    return inner

add_5 = curry_add(5)
print(add_5(3))  # 8

# Recursion (functional style for loops)
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

print(factorial(5))  # 120

# Tail recursion (Python doesn't optimize this)
def tail_factorial(n, accumulator=1):
    if n <= 1:
        return accumulator
    return tail_factorial(n - 1, n * accumulator)

print(tail_factorial(5))  # 120

---

### Q37: What are lambda functions and when should you use them?

**Answer:**
Lambda functions are anonymous, single-expression functions useful for short operations.

```python
# Basic lambda
square = lambda x: x ** 2
print(square(5))  # 25

# Multiple arguments
add = lambda x, y: x + y
print(add(3, 4))  # 7

# With map
numbers = [1, 2, 3, 4, 5]
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# With filter
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4]

# With sorted - custom key
students = [
    {"name": "Alice", "grade": 85},
    {"name": "Bob", "grade": 92},
    {"name": "Charlie", "grade": 78}
]
sorted_students = sorted(students, key=lambda s: s["grade"])
print([s["name"] for s in sorted_students])  # ['Charlie', 'Alice', 'Bob']

# With max/min
oldest = max(students, key=lambda s: s["grade"])
print(oldest)  # {'name': 'Bob', 'grade': 92}

# In list comprehensions with conditional
result = [(lambda x: x ** 2)(x) if x % 2 == 0 else x for x in range(10)]
print(result)  # [0, 1, 4, 3, 16, 5, 36, 7, 64, 9]

# Immediately invoked lambda
result = (lambda x, y: x + y)(5, 3)
print(result)  # 8

# Lambda with default arguments
greet = lambda name, greeting="Hello": f"{greeting}, {name}!"
print(greet("Alice"))  # "Hello, Alice!"
print(greet("Bob", "Hi"))  # "Hi, Bob!"

# When NOT to use lambda:
# ✗ Complex logic (use def instead)
# Bad:
complex_lambda = lambda x: x ** 2 if x > 0 else -x if x < 0 else 0

# Good:
def complex_function(x):
    if x > 0:
        return x ** 2
    elif x < 0:
        return -x
    else:
        return 0

# ✗ Assigning to variable (use def for named functions)
# Bad:
my_func = lambda x: x ** 2

# Good:
def my_func(x):
    return x ** 2

# ✓ Good use cases:
# 1. Short callbacks
button.on_click(lambda: print("Button clicked"))

# 2. Key functions
sorted(words, key=lambda w: len(w))

# 3. One-time transformations
list(map(lambda x: x.strip().lower(), lines))

---

### Q38: How do closures work in Python?

**Answer:**
Closures are functions that remember values from their enclosing scope, even after that scope has finished executing.

```python
# Basic closure
def outer(x):
    def inner(y):
        return x + y  # inner() "closes over" x
    return inner

add_5 = outer(5)
print(add_5(3))  # 8
print(add_5(10))  # 15

# The closure remembers x=5
print(add_5.__closure__)  # Cell objects containing closed-over values

# Multiple closures with different values
add_5 = outer(5)
add_10 = outer(10)
print(add_5(3))   # 8
print(add_10(3))  # 13

# Closure for maintaining state
def counter():
    count = 0
    
    def increment():
        nonlocal count  # Modify variable from enclosing scope
        count += 1
        return count
    
    return increment

counter1 = counter()
print(counter1())  # 1
print(counter1())  # 2
print(counter1())  # 3

counter2 = counter()
print(counter2())  # 1 - independent state

# Closure factory
def multiplier(factor):
    return lambda x: x * factor

double = multiplier(2)
triple = multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15

# Practical example: decorator with arguments
def repeat(times):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(times):
                result = func(*args, **kwargs)
            return result
        return wrapper
    return decorator

@repeat(3)
def greet(name):
    print(f"Hello, {name}!")

greet("Alice")
# Hello, Alice!
# Hello, Alice!
# Hello, Alice!

# Closure for private variables
def create_account(initial_balance):
    balance = initial_balance  # Private variable
    
    def deposit(amount):
        nonlocal balance
        balance += amount
        return balance
    
    def withdraw(amount):
        nonlocal balance
        if amount > balance:
            return "Insufficient funds"
        balance -= amount
        return balance
    
    def get_balance():
        return balance
    
    return {
        "deposit": deposit,
        "withdraw": withdraw,
        "get_balance": get_balance
    }

account = create_account(1000)
print(account["deposit"](500))  # 1500
print(account["withdraw"](200))  # 1300
print(account["get_balance"]())  # 1300
# Can't access balance directly - it's encapsulated!

# Closure with loop - common pitfall
# Wrong:
functions = []
for i in range(3):
    functions.append(lambda: i)  # All capture same i!

print([f() for f in functions])  # [2, 2, 2] - all return 2!

# Correct:
functions = []
for i in range(3):
    functions.append(lambda i=i: i)  # Default argument captures current i

print([f() for f in functions])  # [0, 1, 2]

# Or use closure properly:
def make_printer(i):
    return lambda: i

functions = [make_printer(i) for i in range(3)]
print([f() for f in functions])  # [0, 1, 2]

---

### Q39: What are generators and how do they differ from regular functions?

**Answer:**
Generators produce values lazily using `yield`, saving memory and enabling infinite sequences.

```python
# Regular function - returns all at once
def get_numbers_list(n):
    result = []
    for i in range(n):
        result.append(i)
    return result

# Memory intensive for large n
numbers = get_numbers_list(1000000)  # Creates list of 1M items

# Generator - yields one at a time
def get_numbers_generator(n):
    for i in range(n):
        yield i  # Pauses and returns value

# Memory efficient
numbers = get_numbers_generator(1000000)  # Generator object, no list created
print(next(numbers))  # 0
print(next(numbers))  # 1

# Generator with for loop
for num in get_numbers_generator(5):
    print(num)  # 0, 1, 2, 3, 4

# Generator expression (like list comprehension)
squares = (x ** 2 for x in range(10))  # Generator
squares_list = [x ** 2 for x in range(10)]  # List

print(type(squares))  # <class 'generator'>
print(next(squares))  # 0
print(next(squares))  # 1

# Infinite generator
def infinite_sequence():
    num = 0
    while True:
        yield num
        num += 1

gen = infinite_sequence()
print(next(gen))  # 0
print(next(gen))  # 1
# Can keep going forever!

# Fibonacci generator
def fibonacci():
    a, b = 0, 1
    while True:
        yield a
        a, b = b, a + b

fib = fibonacci()
print([next(fib) for _ in range(10)])
# [0, 1, 1, 2, 3, 5, 8, 13, 21, 34]

# Generator with state
def counter(start=0):
    count = start
    while True:
        val = yield count
        if val is not None:
            count = val
        else:
            count += 1

c = counter()
print(next(c))  # 0
print(next(c))  # 1
print(c.send(10))  # 10 - reset counter
print(next(c))  # 11

# File processing with generator
def read_large_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()

# Memory efficient - one line at a time
# for line in read_large_file('large_file.txt'):
#     process(line)

# Generator pipeline
def numbers(n):
    for i in range(n):
        yield i

def squares(nums):
    for num in nums:
        yield num ** 2

def evens(nums):
    for num in nums:
        if num % 2 == 0:
            yield num

# Chain generators
pipeline = evens(squares(numbers(10)))
print(list(pipeline))  # [0, 4, 16, 36, 64]

# Generator with cleanup
def managed_resource(resource_name):
    print(f"Acquiring {resource_name}")
    try:
        yield resource_name
    finally:
        print(f"Releasing {resource_name}")

gen = managed_resource("database")
resource = next(gen)
print(f"Using {resource}")
# gen.close()  # Triggers finally block

# Performance comparison
import sys

# List uses more memory
list_comp = [x ** 2 for x in range(10000)]
print(f"List size: {sys.getsizeof(list_comp)} bytes")

# Generator uses constant memory
gen_expr = (x ** 2 for x in range(10000))
print(f"Generator size: {sys.getsizeof(gen_expr)} bytes")
# Generator size is tiny regardless of sequence length!

---

### Q40: How do you use `map()`, `filter()`, and `reduce()` effectively?

**Answer:**
These functional programming tools transform and combine data without explicit loops.

```python
from functools import reduce

# MAP - Apply function to each element
numbers = [1, 2, 3, 4, 5]

# With lambda
squared = list(map(lambda x: x ** 2, numbers))
print(squared)  # [1, 4, 9, 16, 25]

# With named function
def cube(x):
    return x ** 3

cubed = list(map(cube, numbers))
print(cubed)  # [1, 8, 27, 64, 125]

# Multiple iterables
list1 = [1, 2, 3]
list2 = [10, 20, 30]
result = list(map(lambda x, y: x + y, list1, list2))
print(result)  # [11, 22, 33]

# Map with strings
words = ["hello", "world", "python"]
upper_words = list(map(str.upper, words))
print(upper_words)  # ['HELLO', 'WORLD', 'PYTHON']

# FILTER - Keep elements that pass test
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Get even numbers
evens = list(filter(lambda x: x % 2 == 0, numbers))
print(evens)  # [2, 4, 6, 8, 10]

# Get numbers > 5
greater_than_5 = list(filter(lambda x: x > 5, numbers))
print(greater_than_5)  # [6, 7, 8, 9, 10]

# Filter strings
words = ["apple", "banana", "avocado", "cherry", "apricot"]
a_words = list(filter(lambda w: w.startswith('a'), words))
print(a_words)  # ['apple', 'avocado', 'apricot']

# Filter with None - removes falsy values
mixed = [0, 1, False, True, "", "hello", None, [], [1, 2]]
truthy = list(filter(None, mixed))
print(truthy)  # [1, True, 'hello', [1, 2]]

# REDUCE - Combine elements into single value
numbers = [1, 2, 3, 4, 5]

# Sum all numbers
total = reduce(lambda x, y: x + y, numbers)
print(total)  # 15

# Product of all numbers
product = reduce(lambda x, y: x * y, numbers)
print(product)  # 120

# Find maximum
maximum = reduce(lambda x, y: x if x > y else y, numbers)
print(maximum)  # 5

# With initial value
total_with_init = reduce(lambda x, y: x + y, numbers, 100)
print(total_with_init)  # 115 (100 + sum of numbers)

# Complex reduce - flatten nested list
nested = [[1, 2], [3, 4], [5, 6]]
flattened = reduce(lambda x, y: x + y, nested)
print(flattened)  # [1, 2, 3, 4, 5, 6]

# Reduce for string concatenation
words = ["Hello", "World", "Python"]
sentence = reduce(lambda x, y: x + " " + y, words)
print(sentence)  # "Hello World Python"

# COMBINING map, filter, reduce
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Square even numbers, then sum them
result = reduce(
    lambda x, y: x + y,
    map(
        lambda x: x ** 2,
        filter(lambda x: x % 2 == 0, numbers)
    )
)
print(result)  # 220 (4 + 16 + 36 + 64 + 100)

# Modern alternative: list comprehension (often more readable)
result = sum(x ** 2 for x in numbers if x % 2 == 0)
print(result)  # 220

# Real-world examples

# 1. Data transformation
users = [
    {"name": "Alice", "age": 30},
    {"name": "Bob", "age": 25},
    {"name": "Charlie", "age": 35}
]

# Get names of users over 26
names = list(map(
    lambda u: u["name"],
    filter(lambda u: u["age"] > 26, users)
))
print(names)  # ['Alice', 'Charlie']

# 2. Calculate total price with discount
prices = [10.00, 20.00, 30.00, 40.00]
discount = 0.1

total = reduce(
    lambda x, y: x + y,
    map(lambda p: p * (1 - discount), prices)
)
print(f"Total: ${total:.2f}")  # Total: $90.00

# 3. Parse and validate data
raw_data = ["  123  ", "456", "  789  ", "abc", "012"]

# Strip, filter numeric, convert to int
valid_numbers = list(map(
    int,
    filter(
        lambda x: x.isdigit(),
        map(str.strip, raw_data)
    )
))
print(valid_numbers)  # [123, 456, 789, 12]

# When to use what:
# • map: Transform each element
# • filter: Select elements based on condition
# • reduce: Combine all elements into one value
# • List comprehension: Often more Pythonic and readable

---

### Q41: What is the `functools` module and what are its key functions?

**Answer:**
`functools` provides higher-order functions and operations on callable objects.

```python
from functools import (
    lru_cache, cache, partial, reduce, wraps,
    total_ordering, singledispatch, cached_property
)

# 1. lru_cache - Memoization with Least Recently Used cache
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

print(fibonacci(100))  # Fast due to caching
print(fibonacci.cache_info())  # CacheInfo(hits=..., misses=..., maxsize=128, currsize=...)

# Clear cache
fibonacci.cache_clear()

# 2. cache - Unlimited cache (Python 3.9+)
@cache
def expensive_function(x):
    print(f"Computing for {x}")
    return x ** 2

print(expensive_function(5))  # "Computing for 5", then 25
print(expensive_function(5))  # 25 (cached, no print)

# 3. partial - Pre-fill function arguments
def power(base, exponent):
    return base ** exponent

square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))    # 125

# Practical partial use
from functools import partial
import re

# Create specialized regex functions
find_emails = partial(re.findall, r'\b[\w.-]+@[\w.-]+\.\w+\b')
find_urls = partial(re.findall, r'https?://[\w./]+')

text = "Contact me at user@example.com or visit https://example.com"
print(find_emails(text))
print(find_urls(text))

# 4. wraps - Preserve function metadata in decorators
def my_decorator(func):
    @wraps(func)  # Without this, __name__, __doc__ would be lost
    def wrapper(*args, **kwargs):
        print("Before")
        result = func(*args, **kwargs)
        print("After")
        return result
    return wrapper

@my_decorator
def greet(name):
    """Greet someone"""
    return f"Hello, {name}"

print(greet.__name__)  # "greet" (not "wrapper")
print(greet.__doc__)   # "Greet someone"

# 5. reduce - Combine iterable elements
numbers = [1, 2, 3, 4, 5]
sum_all = reduce(lambda x, y: x + y, numbers)
print(sum_all)  # 15

# 6. total_ordering - Generate comparison methods
@total_ordering
class Student:
    def __init__(self, name, grade):
        self.name = name
        self.grade = grade
    
    def __eq__(self, other):
        return self.grade == other.grade
    
    def __lt__(self, other):
        return self.grade < other.grade
    # __le__, __gt__, __ge__ generated automatically!

s1 = Student("Alice", 85)
s2 = Student("Bob", 90)
print(s1 < s2)   # True
print(s1 <= s2)  # True (generated)
print(s1 > s2)   # False (generated)

# 7. singledispatch - Function overloading by type
@singledispatch
def process(arg):
    print(f"Default: {arg}")

@process.register(int)
def _(arg):
    print(f"Processing integer: {arg * 2}")

@process.register(str)
def _(arg):
    print(f"Processing string: {arg.upper()}")

@process.register(list)
def _(arg):
    print(f"Processing list: {len(arg)} items")

process(5)          # "Processing integer: 10"
process("hello")    # "Processing string: HELLO"
process([1, 2, 3])  # "Processing list: 3 items"
process(3.14)       # "Default: 3.14"

# 8. cached_property - Computed once, cached
class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    @cached_property
    def expensive_computation(self):
        print("Computing...")
        return sum(x ** 2 for x in self.data)

processor = DataProcessor([1, 2, 3, 4, 5])
print(processor.expensive_computation)  # "Computing..." then 55
print(processor.expensive_computation)  # 55 (cached)

---

### Q42: How do you use `itertools` for efficient iteration?

**Answer:**
`itertools` provides memory-efficient tools for creating and working with iterators.

```python
import itertools

# 1. count - Infinite counter
counter = itertools.count(start=10, step=2)
print(next(counter))  # 10
print(next(counter))  # 12
print(next(counter))  # 14

# Use with zip to limit
for i, value in zip(itertools.count(), ['a', 'b', 'c']):
    print(i, value)  # 0 a, 1 b, 2 c

# 2. cycle - Infinite cycle through iterable
colors = itertools.cycle(['red', 'green', 'blue'])
print([next(colors) for _ in range(5)])
# ['red', 'green', 'blue', 'red', 'green']

# 3. repeat - Repeat value
repeated = itertools.repeat('A', times=3)
print(list(repeated))  # ['A', 'A', 'A']

# Useful with map
result = list(map(pow, range(5), itertools.repeat(2)))
print(result)  # [0, 1, 4, 9, 16] - squares

# 4. chain - Combine multiple iterables
list1 = [1, 2, 3]
list2 = [4, 5, 6]
combined = itertools.chain(list1, list2)
print(list(combined))  # [1, 2, 3, 4, 5, 6]

# chain.from_iterable - flatten nested structure
nested = [[1, 2], [3, 4], [5, 6]]
flat = itertools.chain.from_iterable(nested)
print(list(flat))  # [1, 2, 3, 4, 5, 6]

# 5. islice - Slice an iterator
numbers = itertools.count()
first_10 = list(itertools.islice(numbers, 10))
print(first_10)  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

# Skip and take
numbers = range(100)
result = list(itertools.islice(numbers, 10, 20, 2))
print(result)  # [10, 12, 14, 16, 18]

# 6. combinations - All combinations
items = ['A', 'B', 'C']
combos = itertools.combinations(items, 2)
print(list(combos))  # [('A', 'B'), ('A', 'C'), ('B', 'C')]

# 7. permutations - All permutations
items = ['A', 'B', 'C']
perms = itertools.permutations(items, 2)
print(list(perms))
# [('A', 'B'), ('A', 'C'), ('B', 'A'), ('B', 'C'), ('C', 'A'), ('C', 'B')]

# 8. product - Cartesian product
colors = ['red', 'blue']
sizes = ['S', 'M', 'L']
products = itertools.product(colors, sizes)
print(list(products))
# [('red', 'S'), ('red', 'M'), ('red', 'L'), 
#  ('blue', 'S'), ('blue', 'M'), ('blue', 'L')]

# Equivalent to nested loops
for color in colors:
    for size in sizes:
        print(color, size)

# 9. groupby - Group consecutive elements
data = [1, 1, 2, 2, 2, 3, 3, 1, 1]
for key, group in itertools.groupby(data):
    print(key, list(group))
# 1 [1, 1]
# 2 [2, 2, 2]
# 3 [3, 3]
# 1 [1, 1]

# With key function
words = ['apple', 'apricot', 'banana', 'berry', 'cherry']
for key, group in itertools.groupby(words, key=lambda w: w[0]):
    print(f"{key}: {list(group)}")
# a: ['apple', 'apricot']
# b: ['banana', 'berry']
# c: ['cherry']

# 10. takewhile / dropwhile - Take/drop while condition true
numbers = [1, 2, 3, 4, 5, 1, 2, 3]
result = list(itertools.takewhile(lambda x: x < 4, numbers))
print(result)  # [1, 2, 3]

result = list(itertools.dropwhile(lambda x: x < 4, numbers))
print(result)  # [4, 5, 1, 2, 3]

# 11. filterfalse - Opposite of filter
numbers = [1, 2, 3, 4, 5, 6]
odds = list(itertools.filterfalse(lambda x: x % 2 == 0, numbers))
print(odds)  # [1, 3, 5]

# 12. accumulate - Running totals
numbers = [1, 2, 3, 4, 5]
running_sum = list(itertools.accumulate(numbers))
print(running_sum)  # [1, 3, 6, 10, 15]

# With custom operation
running_product = list(itertools.accumulate(numbers, lambda x, y: x * y))
print(running_product)  # [1, 2, 6, 24, 120]

# 13. zip_longest - Zip with fill value
from itertools import zip_longest

list1 = [1, 2, 3]
list2 = ['a', 'b', 'c', 'd', 'e']
result = list(zip_longest(list1, list2, fillvalue=0))
print(result)  # [(1, 'a'), (2, 'b'), (3, 'c'), (0, 'd'), (0, 'e')]

# Real-world examples

# Pagination
def paginate(items, page_size):
    args = [iter(items)] * page_size
    return itertools.zip_longest(*args, fillvalue=None)

data = range(10)
for page in paginate(data, 3):
    print(list(filter(None, page)))  # Remove None values

# Moving window
def sliding_window(iterable, n):
    iterators = itertools.tee(iterable, n)
    for i, it in enumerate(iterators):
        for _ in range(i):
            next(it, None)
    return zip(*iterators)

numbers = [1, 2, 3, 4, 5]
for window in sliding_window(numbers, 3):
    print(window)
# (1, 2, 3)
# (2, 3, 4)
# (3, 4, 5)

---

### Q43: What are comprehensions and how do you use them effectively?

**Answer:**
Comprehensions provide concise syntax for creating lists, sets, dicts, and generators.

```python
# List comprehension
squares = [x ** 2 for x in range(10)]
print(squares)  # [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]

# With condition
evens = [x for x in range(10) if x % 2 == 0]
print(evens)  # [0, 2, 4, 6, 8]

# With if-else (ternary)
result = [x if x % 2 == 0 else -x for x in range(10)]
print(result)  # [0, -1, 2, -3, 4, -5, 6, -7, 8, -9]

# Nested loops
pairs = [(x, y) for x in range(3) for y in range(3)]
print(pairs)
# [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]

# Flatten nested list
nested = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
flat = [item for sublist in nested for item in sublist]
print(flat)  # [1, 2, 3, 4, 5, 6, 7, 8, 9]

# Set comprehension - removes duplicates
text = "hello world"
unique_chars = {char for char in text if char.isalpha()}
print(unique_chars)  # {'h', 'e', 'l', 'o', 'w', 'r', 'd'}

# Dictionary comprehension
word_lengths = {word: len(word) for word in ['hello', 'world', 'python']}
print(word_lengths)  # {'hello': 5, 'world': 5, 'python': 6}

# Swap keys and values
original = {'a': 1, 'b': 2, 'c': 3}
swapped = {v: k for k, v in original.items()}
print(swapped)  # {1: 'a', 2: 'b', 3: 'c'}

# With condition
prices = {'apple': 0.50, 'banana': 0.30, 'cherry': 0.80}
expensive = {item: price for item, price in prices.items() if price > 0.40}
print(expensive)  # {'apple': 0.5, 'cherry': 0.8}

# Generator expression - memory efficient
squares_gen = (x ** 2 for x in range(1000000))  # No list created!
print(next(squares_gen))  # 0
print(next(squares_gen))  # 1

# Use in functions that accept iterables
total = sum(x ** 2 for x in range(100))  # Generator, not list
print(total)  # 328350

# Real-world examples

# 1. Parse CSV data
csv_lines = ["name,age,city", "Alice,30,NYC", "Bob,25,LA"]
data = [line.split(',') for line in csv_lines[1:]]  # Skip header
print(data)  # [['Alice', '30', 'NYC'], ['Bob', '25', 'LA']]

# 2. Filter and transform
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
doubled_evens = [x * 2 for x in numbers if x % 2 == 0]
print(doubled_evens)  # [4, 8, 12, 16, 20]

# 3. Create lookup dictionary
users = [
    {'id': 1, 'name': 'Alice'},
    {'id': 2, 'name': 'Bob'},
    {'id': 3, 'name': 'Charlie'}
]
user_lookup = {user['id']: user['name'] for user in users}
print(user_lookup)  # {1: 'Alice', 2: 'Bob', 3: 'Charlie'}

# 4. Matrix operations
matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
# Transpose
transposed = [[row[i] for row in matrix] for i in range(len(matrix[0]))]
print(transposed)  # [[1, 4, 7], [2, 5, 8], [3, 6, 9]]

# 5. Filtering dictionary
users = {
    'alice': {'age': 30, 'active': True},
    'bob': {'age': 25, 'active': False},
    'charlie': {'age': 35, 'active': True}
}
active_users = {name: data for name, data in users.items() if data['active']}
print(active_users)

# When NOT to use comprehensions
# Too complex - hurts readability
# Bad:
# result = [process(x) if condition1(x) else alternative(x) 
#           for x in items if condition2(x) and condition3(x)]

# Better as loop:
result = []
for x in items:
    if condition2(x) and condition3(x):
        if condition1(x):
            result.append(process(x))
        else:
            result.append(alternative(x))

# Performance tip: Use generators for large datasets
# Memory heavy:
squares_list = [x ** 2 for x in range(10000000)]  # Creates huge list

# Memory efficient:
squares_gen = (x ** 2 for x in range(10000000))  # Generator object
total = sum(squares_gen)  # Process one at a time

---

### Q44: How do you work with variable-length argument lists?

**Answer:**
Use `*args` for positional arguments and `**kwargs` for keyword arguments to accept variable-length inputs.

```python
# *args - variable positional arguments
def sum_all(*args):
    return sum(args)

print(sum_all(1, 2, 3))  # 6
print(sum_all(1, 2, 3, 4, 5))  # 15

# **kwargs - variable keyword arguments
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Alice", age=30, city="NYC")
# name: Alice
# age: 30
# city: NYC

# Combining regular args, *args, and **kwargs
def complex_function(required, *args, optional=None, **kwargs):
    print(f"Required: {required}")
    print(f"Args: {args}")
    print(f"Optional: {optional}")
    print(f"Kwargs: {kwargs}")

complex_function("must have", 1, 2, 3, optional="yes", extra="data", more="info")
# Required: must have
# Args: (1, 2, 3)
# Optional: yes
# Kwargs: {'extra': 'data', 'more': 'info'}

# Unpacking arguments
def add(a, b, c):
    return a + b + c

numbers = [1, 2, 3]
print(add(*numbers))  # Unpacks list as arguments: add(1, 2, 3)

params = {'a': 1, 'b': 2, 'c': 3}
print(add(**params))  # Unpacks dict as keyword arguments

# Forwarding arguments to another function
def wrapper(*args, **kwargs):
    print("Wrapper called")
    return original_function(*args, **kwargs)

def original_function(x, y, z=10):
    return x + y + z

print(wrapper(1, 2))  # 13
print(wrapper(1, 2, z=20))  # 23

# Keyword-only arguments (after *)
def create_user(name, *, email, age):
    # email and age MUST be passed as keyword arguments
    return {'name': name, 'email': email, 'age': age}

user = create_user("Alice", email="alice@example.com", age=30)
# create_user("Alice", "alice@example.com", 30)  # Error!

# Positional-only arguments (before /) - Python 3.8+
def greet(name, /, greeting="Hello"):
    # name must be positional
    return f"{greeting}, {name}!"

print(greet("Alice"))  # OK
print(greet("Alice", greeting="Hi"))  # OK
# print(greet(name="Alice"))  # Error!

# Combining positional-only, normal, and keyword-only
def func(pos_only, /, pos_or_kwd, *, kwd_only):
    print(f"Positional only: {pos_only}")
    print(f"Positional or keyword: {pos_or_kwd}")
    print(f"Keyword only: {kwd_only}")

func(1, 2, kwd_only=3)  # OK
func(1, pos_or_kwd=2, kwd_only=3)  # OK
# func(pos_only=1, pos_or_kwd=2, kwd_only=3)  # Error!

# Practical examples

# 1. Flexible logging function
def log(level, message, *args, **kwargs):
    formatted_message = message.format(*args)
    metadata = " ".join(f"{k}={v}" for k, v in kwargs.items())
    print(f"[{level}] {formatted_message} {metadata}")

log("INFO", "User {} logged in", "Alice", timestamp="2024-01-15", ip="192.168.1.1")
# [INFO] User Alice logged in timestamp=2024-01-15 ip=192.168.1.1

# 2. Function composition
def compose(*functions):
    def inner(x):
        for func in reversed(functions):
            x = func(x)
        return x
    return inner

def add_one(x):
    return x + 1

def double(x):
    return x * 2

def square(x):
    return x ** 2

combined = compose(square, double, add_one)
print(combined(5))  # ((5 + 1) * 2) ** 2 = 144

# 3. Partial application with *args and **kwargs
def partial_right(func, *fixed_args):
    def wrapper(*args):
        return func(*args, *fixed_args)
    return wrapper

def power(base, exponent):
    return base ** exponent

square_func = partial_right(power, 2)
print(square_func(5))  # 25

---

### Q45: What are type hints and how do you use them?

**Answer:**
Type hints (Python 3.5+) provide optional static typing for better code documentation and tooling support.

```python
from typing import List, Dict, Tuple, Optional, Union, Any, Callable

# Basic type hints
def greet(name: str) -> str:
    return f"Hello, {name}!"

def add(a: int, b: int) -> int:
    return a + b

# Collection types
def process_items(items: List[int]) -> int:
    return sum(items)

def get_user_data() -> Dict[str, Union[str, int]]:
    return {"name": "Alice", "age": 30}

def get_coordinates() -> Tuple[float, float]:
    return (10.5, 20.3)

# Optional - value or None
def find_user(user_id: int) -> Optional[str]:
    users = {1: "Alice", 2: "Bob"}
    return users.get(user_id)  # Returns str or None

# Union - multiple types
def process_data(data: Union[int, str, List[int]]) -> str:
    if isinstance(data, int):
        return f"Integer: {data}"
    elif isinstance(data, str):
        return f"String: {data}"
    else:
        return f"List: {data}"

# Any - any type (use sparingly)
def flexible_function(param: Any) -> Any:
    return param

# Callable - function type
def apply_function(func: Callable[[int, int], int], a: int, b: int) -> int:
    return func(a, b)

def multiply(x: int, y: int) -> int:
    return x * y

result = apply_function(multiply, 5, 3)

# Type aliases
UserId = int
UserName = str
UserData = Dict[str, Union[str, int]]

def create_user(user_id: UserId, name: UserName) -> UserData:
    return {"id": user_id, "name": name}

# Generic types
from typing import TypeVar, Generic

T = TypeVar('T')

class Stack(Generic[T]):
    def __init__(self) -> None:
        self.items: List[T] = []
    
    def push(self, item: T) -> None:
        self.items.append(item)
    
    def pop(self) -> T:
        return self.items.pop()

# Type-specific stacks
int_stack: Stack[int] = Stack()
int_stack.push(1)
int_stack.push(2)

str_stack: Stack[str] = Stack()
str_stack.push("hello")

# Class type hints
class Person:
    def __init__(self, name: str, age: int) -> None:
        self.name: str = name
        self.age: int = age
    
    def greet(self) -> str:
        return f"Hello, I'm {self.name}"

# Forward references (self-referencing types)
class Node:
    def __init__(self, value: int, next: Optional['Node'] = None) -> None:
        self.value = value
        self.next = next

# Python 3.10+ - Union with | operator
def process(value: int | str) -> str:
    return str(value)

# Python 3.9+ - Built-in generics
def get_items() -> list[str]:  # No need for typing.List
    return ["a", "b", "c"]

def get_mapping() -> dict[str, int]:
    return {"a": 1, "b": 2}

# Literal types - specific values only
from typing import Literal

def set_mode(mode: Literal["read", "write", "append"]) -> None:
    print(f"Mode: {mode}")

set_mode("read")  # OK
# set_mode("delete")  # Type checker warns!

# Type checking with runtime validation
def validate_age(age: int) -> None:
    if not isinstance(age, int):
        raise TypeError("Age must be an integer")
    if age < 0 or age > 150:
        raise ValueError("Invalid age")

# Using mypy for static type checking
# Run: mypy your_script.py

# Example that mypy would catch:
def add_numbers(a: int, b: int) -> int:
    return a + b

# result: str = add_numbers(5, 3)  # mypy error: incompatible types

# TypedDict for structured dictionaries (Python 3.8+)
from typing import TypedDict

class UserDict(TypedDict):
    name: str
    age: int
    email: str

def process_user(user: UserDict) -> str:
    return f"{user['name']} ({user['age']})"

# Protocol for structural subtyping
from typing import Protocol

class Drawable(Protocol):
    def draw(self) -> str:
        ...

def render(obj: Drawable) -> None:
    print(obj.draw())

class Circle:
    def draw(self) -> str:
        return "Drawing circle"

render(Circle())  # Type checker approves!

# Benefits of type hints:
# 1. Better IDE autocomplete and error detection
# 2. Self-documenting code
# 3. Catch bugs before runtime
# 4. Easier refactoring
# 5. Better collaboration in teams

---

### Q46: How do you use Python's `operator` module?

**Answer:**
The `operator` module provides function equivalents for Python's operators, useful in functional programming.

```python
import operator

# Arithmetic operators
print(operator.add(5, 3))  # 8 (same as 5 + 3)
print(operator.sub(5, 3))  # 2 (same as 5 - 3)
print(operator.mul(5, 3))  # 15 (same as 5 * 3)
print(operator.truediv(10, 2))  # 5.0 (same as 10 / 2)
print(operator.floordiv(10, 3))  # 3 (same as 10 // 3)
print(operator.mod(10, 3))  # 1 (same as 10 % 3)
print(operator.pow(2, 3))  # 8 (same as 2 ** 3)

# Comparison operators
print(operator.eq(5, 5))  # True (same as 5 == 5)
print(operator.ne(5, 3))  # True (same as 5 != 3)
print(operator.lt(3, 5))  # True (same as 3 < 5)
print(operator.le(5, 5))  # True (same as 5 <= 5)
print(operator.gt(5, 3))  # True (same as 5 > 3)
print(operator.ge(5, 5))  # True (same as 5 >= 5)

# Logical operators
print(operator.and_(True, False))  # False (same as True and False)
print(operator.or_(True, False))  # True (same as True or False)
print(operator.not_(True))  # False (same as not True)

# Using with higher-order functions
numbers = [1, 2, 3, 4, 5]

# Instead of lambda
from functools import reduce
total = reduce(operator.add, numbers)
print(total)  # 15

# Sort by specific field
students = [
    {'name': 'Alice', 'grade': 85},
    {'name': 'Bob', 'grade': 92},
    {'name': 'Charlie', 'grade': 78}
]

# Using operator.itemgetter (better than lambda)
sorted_students = sorted(students, key=operator.itemgetter('grade'))
print([s['name'] for s in sorted_students])  # ['Charlie', 'Alice', 'Bob']

# Multiple keys
data = [
    ('Alice', 30, 'NYC'),
    ('Bob', 25, 'LA'),
    ('Alice', 25, 'Boston')
]
sorted_data = sorted(data, key=operator.itemgetter(0, 1))
print(sorted_data)
# [('Alice', 25, 'Boston'), ('Alice', 30, 'NYC'), ('Bob', 25, 'LA')]

# operator.attrgetter - get attributes
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

people = [Person('Alice', 30), Person('Bob', 25), Person('Charlie', 35)]
sorted_people = sorted(people, key=operator.attrgetter('age'))
print([p.name for p in sorted_people])  # ['Bob', 'Alice', 'Charlie']

# Multiple attributes
sorted_people = sorted(people, key=operator.attrgetter('name', 'age'))

# operator.methodcaller - call method
strings = ['hello', 'world', 'python']
upper_strings = list(map(operator.methodcaller('upper'), strings))
print(upper_strings)  # ['HELLO', 'WORLD', 'PYTHON']

# With arguments
strings = ['  hello  ', '  world  ']
trimmed = list(map(operator.methodcaller('strip'), strings))
print(trimmed)  # ['hello', 'world']

# Replace value
texts = ['hello world', 'foo bar']
replaced = list(map(operator.methodcaller('replace', 'o', '0'), texts))
print(replaced)  # ['hell0 w0rld', 'f00 bar']

# In-place operators
numbers = [1, 2, 3]
operator.iadd(numbers, [4, 5])  # Same as numbers += [4, 5]
print(numbers)  # [1, 2, 3, 4, 5]

# Practical examples

# 1. Max/min by specific field
products = [
    {'name': 'Apple', 'price': 1.50},
    {'name': 'Banana', 'price': 0.75},
    {'name': 'Cherry', 'price': 2.00}
]

cheapest = min(products, key=operator.itemgetter('price'))
print(cheapest)  # {'name': 'Banana', 'price': 0.75}

# 2. Group by field
from itertools import groupby

data = [
    {'category': 'fruit', 'name': 'apple'},
    {'category': 'fruit', 'name': 'banana'},
    {'category': 'vegetable', 'name': 'carrot'},
    {'category': 'vegetable', 'name': 'lettuce'}
]

data_sorted = sorted(data, key=operator.itemgetter('category'))
for category, items in groupby(data_sorted, key=operator.itemgetter('category')):
    print(f"{category}: {[item['name'] for item in items]}")

# 3. Chaining comparisons
def compare_tuples(a, b):
    return operator.eq(a, b)

print(compare_tuples((1, 2), (1, 2)))  # True

---

### Q47: What are first-class functions and how do you leverage them?

**Answer:**
First-class functions can be assigned to variables, passed as arguments, and returned from functions.

```python
# Functions as variables
def greet(name):
    return f"Hello, {name}!"

# Assign to variable
say_hello = greet
print(say_hello("Alice"))  # "Hello, Alice!"

# Store in data structures
def add(a, b):
    return a + b

def subtract(a, b):
    return a - b

def multiply(a, b):
    return a * b

operations = {
    'add': add,
    'subtract': subtract,
    'multiply': multiply
}

result = operations['add'](5, 3)
print(result)  # 8

# Functions as arguments (higher-order functions)
def apply_operation(func, a, b):
    return func(a, b)

print(apply_operation(add, 10, 5))  # 15
print(apply_operation(multiply, 10, 5))  # 50

# Functions as return values
def create_multiplier(factor):
    def multiplier(x):
        return x * factor
    return multiplier

double = create_multiplier(2)
triple = create_multiplier(3)

print(double(5))  # 10
print(triple(5))  # 15

# Practical example: Strategy pattern
class PaymentProcessor:
    def __init__(self, payment_method):
        self.payment_method = payment_method
    
    def process(self, amount):
        return self.payment_method(amount)

def credit_card_payment(amount):
    return f"Paid ${amount} with credit card"

def paypal_payment(amount):
    return f"Paid ${amount} with PayPal"

processor = PaymentProcessor(credit_card_payment)
print(processor.process(100))

processor.payment_method = paypal_payment
print(processor.process(50))

# Function factory with configuration
def create_validator(min_val, max_val):
    def validator(value):
        return min_val <= value <= max_val
    return validator

age_validator = create_validator(0, 150)
temperature_validator = create_validator(-273, 1000)

print(age_validator(30))  # True
print(age_validator(200))  # False
print(temperature_validator(25))  # True

# Callback functions
def process_data(data, callback):
    result = [x * 2 for x in data]
    callback(result)

def print_result(result):
    print(f"Result: {result}")

process_data([1, 2, 3], print_result)

# Function registry pattern
_handlers = {}

def register_handler(event_type):
    def decorator(func):
        _handlers[event_type] = func
        return func
    return decorator

@register_handler('click')
def handle_click():
    print("Click handled")

@register_handler('submit')
def handle_submit():
    print("Submit handled")

# Dispatch based on event
def dispatch_event(event_type):
    handler = _handlers.get(event_type)
    if handler:
        handler()

dispatch_event('click')  # "Click handled"

---

### Q48: How do you use recursion effectively in Python?

**Answer:**
Recursion solves problems by having functions call themselves with simpler inputs.

```python
# Basic recursion - factorial
def factorial(n):
    if n <= 1:  # Base case
        return 1
    return n * factorial(n - 1)  # Recursive case

print(factorial(5))  # 120

# Fibonacci sequence
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(10))  # 55

# Problem: fibonacci is inefficient (exponential time)
# Solution: memoization
from functools import lru_cache

@lru_cache(maxsize=None)
def fibonacci_cached(n):
    if n <= 1:
        return n
    return fibonacci_cached(n - 1) + fibonacci_cached(n - 2)

print(fibonacci_cached(100))  # Fast!

# Tree traversal
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right

def inorder_traversal(node):
    if node is None:
        return []
    
    result = []
    result.extend(inorder_traversal(node.left))
    result.append(node.value)
    result.extend(inorder_traversal(node.right))
    return result

# Build tree
root = TreeNode(1,
    TreeNode(2, TreeNode(4), TreeNode(5)),
    TreeNode(3)
)

print(inorder_traversal(root))  # [4, 2, 5, 1, 3]

# Deep copy with recursion
def deep_copy(obj):
    if isinstance(obj, dict):
        return {k: deep_copy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [deep_copy(item) for item in obj]
    else:
        return obj

original = {'a': [1, 2, {'b': 3}]}
copied = deep_copy(original)
copied['a'][2]['b'] = 999
print(original)  # Unchanged

# Flatten nested structure
def flatten(lst):
    result = []
    for item in lst:
        if isinstance(item, list):
            result.extend(flatten(item))
        else:
            result.append(item)
    return result

nested = [1, [2, [3, 4], 5], 6, [7, 8]]
print(flatten(nested))  # [1, 2, 3, 4, 5, 6, 7, 8]

# Tail recursion (Python doesn't optimize this)
def tail_factorial(n, accumulator=1):
    if n <= 1:
        return accumulator
    return tail_factorial(n - 1, n * accumulator)

print(tail_factorial(5))  # 120

# Recursion depth limit
import sys
print(sys.getrecursionlimit())  # Usually 1000

# Increase if needed (use cautiously)
# sys.setrecursionlimit(10000)

# Iterative alternative (preferred for deep recursion)
def factorial_iterative(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

# When to use recursion:
# ✓ Tree/graph traversal
# ✓ Divide and conquer algorithms
# ✓ Backtracking problems
# ✓ When problem naturally recursive
# ✗ Simple loops (use iteration)
# ✗ Deep recursion (stack overflow risk)
# ✗ Performance-critical code (recursion overhead)

---

### Q49: What is partial application and function currying?

**Answer:**
Partial application fixes some arguments of a function, currying transforms multi-argument functions into chains of single-argument functions.

```python
from functools import partial

# Partial application
def power(base, exponent):
    return base ** exponent

# Create specialized functions
square = partial(power, exponent=2)
cube = partial(power, exponent=3)

print(square(5))  # 25
print(cube(5))    # 125

# Practical partial application
import re

# Create specialized regex functions
find_emails = partial(re.findall, r'\b[\w.-]+@[\w.-]+\.\w+\b')
find_numbers = partial(re.findall, r'\d+')

text = "Contact me at user@example.com or call 555-1234"
print(find_emails(text))  # ['user@example.com']
print(find_numbers(text))  # ['555', '1234']

# Manual currying
def curry_add(a):
    def inner(b):
        return a + b
    return inner

add_5 = curry_add(5)
print(add_5(3))  # 8
print(add_5(10))  # 15

# Curry with multiple arguments
def curry_multiply(a):
    def inner1(b):
        def inner2(c):
            return a * b * c
        return inner2
    return inner1

result = curry_multiply(2)(3)(4)
print(result)  # 24

# Generic currying function
def curry(func):
    def curried(*args):
        if len(args) >= func.__code__.co_argcount:
            return func(*args)
        return lambda *more_args: curried(*(args + more_args))
    return curried

@curry
def add_three(a, b, c):
    return a + b + c

print(add_three(1)(2)(3))  # 6
print(add_three(1, 2)(3))  # 6
print(add_three(1, 2, 3))  # 6

# Real-world example: logging with context
def create_logger(level):
    def logger(module):
        def log(message):
            print(f"[{level}] [{module}] {message}")
        return log
    return logger

info_logger = create_logger("INFO")
auth_logger = info_logger("AUTH")
api_logger = info_logger("API")

auth_logger("User logged in")  # [INFO] [AUTH] User logged in
api_logger("Request received")  # [INFO] [API] Request received

# Partial with operator module
from operator import add, mul

add_10 = partial(add, 10)
print(add_10(5))  # 15

double = partial(mul, 2)
print(double(5))  # 10

---

### Q50: How do you use `zip()` and its advanced patterns?

**Answer:**
`zip()` combines multiple iterables into tuples, enabling parallel iteration and data transformation.

```python
# Basic zip
names = ['Alice', 'Bob', 'Charlie']
ages = [30, 25, 35]

for name, age in zip(names, ages):
    print(f"{name} is {age} years old")

# Create dictionary from two lists
keys = ['name', 'age', 'city']
values = ['Alice', 30, 'NYC']
person = dict(zip(keys, values))
print(person)  # {'name': 'Alice', 'age': 30, 'city': 'NYC'}

# Zip stops at shortest sequence
list1 = [1, 2, 3, 4, 5]
list2 = ['a', 'b', 'c']
print(list(zip(list1, list2)))  # [(1, 'a'), (2, 'b'), (3, 'c')]

# zip_longest - continues to longest
from itertools import zip_longest

result = list(zip_longest(list1, list2, fillvalue='X'))
print(result)  # [(1, 'a'), (2, 'b'), (3, 'c'), (4, 'X'), (5, 'X')]

# Unzipping
pairs = [(1, 'a'), (2, 'b'), (3, 'c')]
numbers, letters = zip(*pairs)  # Unpack with *
print(numbers)  # (1, 2, 3)
print(letters)  # ('a', 'b', 'c')

# Transpose matrix
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
transposed = list(zip(*matrix))
print(transposed)
# [(1, 4, 7), (2, 5, 8), (3, 6, 9)]

# Parallel iteration with enumerate
names = ['Alice', 'Bob', 'Charlie']
scores = [85, 92, 78]

for i, (name, score) in enumerate(zip(names, scores), start=1):
    print(f"{i}. {name}: {score}")

# Comparing adjacent elements
numbers = [1, 3, 2, 5, 4, 6]
for current, next_val in zip(numbers, numbers[1:]):
    print(f"{current} -> {next_val}")

# Sliding window
def sliding_window(lst, size):
    iterators = [iter(lst[i:]) for i in range(size)]
    return zip(*iterators)

numbers = [1, 2, 3, 4, 5]
for window in sliding_window(numbers, 3):
    print(window)
# (1, 2, 3)
# (2, 3, 4)
# (3, 4, 5)

# Merge sorted lists
list1 = [1, 3, 5]
list2 = [2, 4, 6]
merged = sorted(list1 + list2)  # Simple way

# Or with zip for paired operations
for a, b in zip(sorted(list1), sorted(list2)):
    print(f"Pair: {a}, {b}")

---

## Section 4: Iterators, Generators & Memory Optimization (Q51-65)

### Q51: How do you create memory-efficient data pipelines?

**Answer:**
Use generators and iterators to process data one item at a time without loading everything into memory.

```python
# Memory inefficient - loads all into memory
def process_file_bad(filename):
    with open(filename) as f:
        lines = f.readlines()  # All lines in memory!
    
    processed = []
    for line in lines:
        processed.append(line.strip().upper())
    return processed

# Memory efficient - generator pipeline
def read_lines(filename):
    with open(filename) as f:
        for line in f:
            yield line.strip()

def filter_empty(lines):
    for line in lines:
        if line:
            yield line

def to_upper(lines):
    for line in lines:
        yield line.upper()

# Chain generators
def process_file_good(filename):
    pipeline = to_upper(filter_empty(read_lines(filename)))
    return pipeline

# Usage - processes one line at a time
# for line in process_file_good('large_file.txt'):
#     process(line)

# Generator expressions in pipeline
def process_data_pipeline(numbers):
    # Each step is a generator
    step1 = (x * 2 for x in numbers)
    step2 = (x for x in step1 if x > 10)
    step3 = (x ** 2 for x in step2)
    return step3

numbers = range(100)
result = process_data_pipeline(numbers)
print(list(result)[:5])  # Only compute what's needed

# CSV processing without pandas
def read_csv(filename):
    with open(filename) as f:
        header = next(f).strip().split(',')
        for line in f:
            values = line.strip().split(',')
            yield dict(zip(header, values))

def filter_rows(rows, condition):
    for row in rows:
        if condition(row):
            yield row

def transform_rows(rows, transform_func):
    for row in rows:
        yield transform_func(row)

# Example usage
# rows = read_csv('data.csv')
# filtered = filter_rows(rows, lambda r: int(r['age']) > 25)
# transformed = transform_rows(filtered, lambda r: {**r, 'age_group': 'adult'})

# Memory comparison
import sys

# List - all in memory
large_list = [x ** 2 for x in range(1000000)]
print(f"List memory: {sys.getsizeof(large_list)} bytes")

# Generator - constant memory
large_gen = (x ** 2 for x in range(1000000))
print(f"Generator memory: {sys.getsizeof(large_gen)} bytes")

# Chunked processing
def process_in_chunks(iterable, chunk_size):
    chunk = []
    for item in iterable:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

# Process large dataset in manageable chunks
numbers = range(1000)
for chunk in process_in_chunks(numbers, 100):
    # Process 100 items at a time
    result = sum(chunk)
    print(f"Chunk sum: {result}")

---

### Q52: What is the difference between `__iter__` and `__next__`?

**Answer:**
`__iter__()` returns the iterator object itself, `__next__()` returns the next value or raises StopIteration.

```python
# Iterator protocol
class CountDown:
    def __init__(self, start):
        self.current = start
    
    def __iter__(self):
        # Return iterator object (self)
        return self
    
    def __next__(self):
        # Return next value or raise StopIteration
        if self.current <= 0:
            raise StopIteration
        self.current -= 1
        return self.current + 1

countdown = CountDown(5)
for num in countdown:
    print(num)  # 5, 4, 3, 2, 1

# Separate iterator from iterable
class NumberRange:
    def __init__(self, start, end):
        self.start = start
        self.end = end
    
    def __iter__(self):
        return NumberRangeIterator(self.start, self.end)

class NumberRangeIterator:
    def __init__(self, start, end):
        self.current = start
        self.end = end
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.current >= self.end:
            raise StopIteration
        value = self.current
        self.current += 1
        return value

# Can iterate multiple times
num_range = NumberRange(1, 5)
print(list(num_range))  # [1, 2, 3, 4]
print(list(num_range))  # [1, 2, 3, 4] - works again!

# Manual iteration
iterator = iter([1, 2, 3])
print(next(iterator))  # 1
print(next(iterator))  # 2
print(next(iterator))  # 3
# print(next(iterator))  # StopIteration

# With default value
print(next(iterator, "Done"))  # "Done" instead of StopIteration

---

### Q53: How do you use `yield from` for generator delegation?

**Answer:**
`yield from` delegates to a sub-generator, simplifying nested generator patterns.

```python
# Without yield from - manual delegation
def flatten_manual(nested):
    for sublist in nested:
        for item in sublist:
            yield item

# With yield from - cleaner
def flatten(nested):
    for sublist in nested:
        yield from sublist

nested = [[1, 2], [3, 4], [5, 6]]
print(list(flatten(nested)))  # [1, 2, 3, 4, 5, 6]

# Recursive flattening
def deep_flatten(nested):
    for item in nested:
        if isinstance(item, list):
            yield from deep_flatten(item)  # Recursive delegation
        else:
            yield item

deeply_nested = [1, [2, [3, 4], 5], 6, [7, 8]]
print(list(deep_flatten(deeply_nested)))  # [1, 2, 3, 4, 5, 6, 7, 8]

# Tree traversal
class TreeNode:
    def __init__(self, value, children=None):
        self.value = value
        self.children = children or []
    
    def traverse(self):
        yield self.value
        for child in self.children:
            yield from child.traverse()

root = TreeNode(1, [
    TreeNode(2, [TreeNode(4), TreeNode(5)]),
    TreeNode(3)
])

print(list(root.traverse()))  # [1, 2, 4, 5, 3]

# Generator chaining
def read_files(*filenames):
    for filename in filenames:
        yield from read_file(filename)

def read_file(filename):
    with open(filename) as f:
        yield from f

# Processes all files as one stream
# for line in read_files('file1.txt', 'file2.txt'):
#     process(line)

---

### Q54: What are itertools recipes and commonly used patterns?

**Answer:**
The itertools documentation provides powerful combination patterns for common iteration tasks.

```python
import itertools

# Pairwise iteration (Python 3.10+ has built-in)
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

numbers = [1, 2, 3, 4, 5]
for a, b in pairwise(numbers):
    print(f"{a} -> {b}")

# Chunking
def chunked(iterable, n):
    it = iter(iterable)
    while chunk := list(itertools.islice(it, n)):
        yield chunk

for chunk in chunked(range(10), 3):
    print(chunk)
# [0, 1, 2]
# [3, 4, 5]
# [6, 7, 8]
# [9]

# Take n items
def take(n, iterable):
    return list(itertools.islice(iterable, n))

print(take(5, itertools.count()))  # [0, 1, 2, 3, 4]

# First true value
def first_true(iterable, default=None, predicate=None):
    return next(filter(predicate, iterable), default)

numbers = [0, 0, 3, 5, 0]
print(first_true(numbers, predicate=lambda x: x > 2))  # 3

# Unique elements preserving order
def unique_everseen(iterable, key=None):
    seen = set()
    for element in iterable:
        k = element if key is None else key(element)
        if k not in seen:
            seen.add(k)
            yield element

data = [1, 2, 2, 3, 1, 4, 5, 3]
print(list(unique_everseen(data)))  # [1, 2, 3, 4, 5]

# Flatten one level
def flatten_one_level(list_of_lists):
    return itertools.chain.from_iterable(list_of_lists)

nested = [[1, 2], [3, 4], [5]]
print(list(flatten_one_level(nested)))  # [1, 2, 3, 4, 5]

# Partition - split into true/false groups
def partition(predicate, iterable):
    t1, t2 = itertools.tee(iterable)
    return (
        filter(predicate, t1),
        itertools.filterfalse(predicate, t2)
    )

numbers = range(10)
evens, odds = partition(lambda x: x % 2 == 0, numbers)
print(list(evens))  # [0, 2, 4, 6, 8]
print(list(odds))   # [1, 3, 5, 7, 9]

# Sliding window
def sliding_window(iterable, n):
    it = iter(iterable)
    window = list(itertools.islice(it, n))
    if len(window) == n:
        yield tuple(window)
    for item in it:
        window = window[1:] + [item]
        yield tuple(window)

for window in sliding_window(range(6), 3):
    print(window)
# (0, 1, 2)
# (1, 2, 3)
# (2, 3, 4)
# (3, 4, 5)

---

### Q55: How do you implement lazy evaluation in Python?

**Answer:**
Lazy evaluation defers computation until the value is actually needed, saving memory and time.

```python
# Eager evaluation - computes immediately
def eager_range(n):
    result = []
    for i in range(n):
        result.append(i ** 2)  # All computed upfront
    return result

# Lazy evaluation - computes on demand
def lazy_range(n):
    for i in range(n):
        yield i ** 2  # Computed only when requested

# Memory comparison
import sys
eager = eager_range(1000000)  # Large memory footprint
lazy = lazy_range(1000000)    # Tiny memory footprint

print(f"Eager: {sys.getsizeof(eager)} bytes")
print(f"Lazy: {sys.getsizeof(lazy)} bytes")

# Lazy property evaluation
class ExpensiveComputation:
    def __init__(self, data):
        self._data = data
        self._result = None
    
    @property
    def result(self):
        if self._result is None:
            print("Computing result...")
            self._result = sum(x ** 2 for x in self._data)
        return self._result

obj = ExpensiveComputation([1, 2, 3, 4, 5])
# No computation yet
print("Created object")
print(obj.result)  # "Computing result..." then 55
print(obj.result)  # 55 (cached)

# Lazy sequences with itertools
import itertools

# Infinite lazy sequences
naturals = itertools.count(1)  # 1, 2, 3, ...
squares = (x ** 2 for x in naturals)  # 1, 4, 9, ...

# Only computed when requested
print(next(squares))  # 1
print(next(squares))  # 4

# Short-circuit evaluation
def expensive_check():
    print("Expensive check called")
    return True

# 'and' short-circuits if first is False
if False and expensive_check():
    pass  # expensive_check() never called

# 'or' short-circuits if first is True
if True or expensive_check():
    pass  # expensive_check() never called

# Lazy data loading
class LazyDataLoader:
    def __init__(self, filename):
        self.filename = filename
        self._data = None
    
    @property
    def data(self):
        if self._data is None:
            print(f"Loading {self.filename}...")
            # Simulate expensive load
            self._data = [1, 2, 3, 4, 5]
        return self._data

loader = LazyDataLoader("data.csv")
# File not loaded yet
print("Loader created")
print(loader.data)  # "Loading data.csv..." then data
print(loader.data)  # Just data (cached)

# Lazy decorator
def lazy_property(func):
    attr_name = '_lazy_' + func.__name__
    
    @property
    def wrapper(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, func(self))
        return getattr(self, attr_name)
    return wrapper

class DataAnalyzer:
    def __init__(self, data):
        self.data = data
    
    @lazy_property
    def mean(self):
        print("Computing mean...")
        return sum(self.data) / len(self.data)
    
    @lazy_property
    def variance(self):
        print("Computing variance...")
        mean = self.mean
        return sum((x - mean) ** 2 for x in self.data) / len(self.data)

analyzer = DataAnalyzer([1, 2, 3, 4, 5])
print(analyzer.mean)      # "Computing mean..." then 3.0
print(analyzer.mean)      # 3.0 (cached)
print(analyzer.variance)  # "Computing variance..." then 2.0

---

### Q56: How do you handle large files without loading them into memory?

**Answer:**
Use iterators and generators to process files line-by-line or in chunks.

```python
# Bad - loads entire file into memory
def process_file_bad(filename):
    with open(filename) as f:
        content = f.read()  # Entire file in memory!
    return content.upper()

# Good - line-by-line processing
def process_file_good(filename):
    with open(filename) as f:
        for line in f:  # Iterator - one line at a time
            yield line.strip().upper()

# Binary file - chunk processing
def read_in_chunks(filename, chunk_size=1024):
    with open(filename, 'rb') as f:
        while chunk := f.read(chunk_size):
            yield chunk

# for chunk in read_in_chunks('large_file.bin'):
#     process(chunk)

# CSV processing without pandas
def process_csv(filename):
    with open(filename) as f:
        header = next(f).strip().split(',')
        for line in f:
            values = line.strip().split(',')
            yield dict(zip(header, values))

# for row in process_csv('data.csv'):
#     if int(row['age']) > 25:
#         print(row['name'])

# Log file analysis
def analyze_log(filename, pattern):
    import re
    with open(filename) as f:
        for line in f:
            if re.search(pattern, line):
                yield line.strip()

# errors = analyze_log('app.log', r'ERROR')
# for error in errors:
#     handle_error(error)

# Counting without loading
def count_lines(filename):
    count = 0
    with open(filename) as f:
        for _ in f:
            count += 1
    return count

# More efficient for very large files
def count_lines_buffered(filename):
    with open(filename, 'rb') as f:
        count = sum(chunk.count(b'\n') for chunk in iter(lambda: f.read(1024 * 1024), b''))
    return count

# Search and replace streaming
def search_and_replace(input_file, output_file, search, replace):
    with open(input_file) as infile, open(output_file, 'w') as outfile:
        for line in infile:
            outfile.write(line.replace(search, replace))

# Merge sorted files
def merge_sorted_files(*filenames):
    import heapq
    
    def file_iterator(filename):
        with open(filename) as f:
            for line in f:
                yield int(line.strip())
    
    iterators = [file_iterator(fn) for fn in filenames]
    for value in heapq.merge(*iterators):
        yield value

# Parallel processing of large file
from multiprocessing import Pool

def process_chunk(chunk):
    # Process chunk of lines
    return [line.upper() for line in chunk]

def parallel_process_file(filename, num_workers=4):
    with open(filename) as f:
        lines = f.readlines()
    
    chunk_size = len(lines) // num_workers
    chunks = [lines[i:i + chunk_size] for i in range(0, len(lines), chunk_size)]
    
    with Pool(num_workers) as pool:
        results = pool.map(process_chunk, chunks)
    
    return [item for chunk in results for item in chunk]

---

### Q57: What are generator-based coroutines and how do they work?

**Answer:**
Generator-based coroutines use `yield` to both produce and consume values, enabling bidirectional communication.

```python
# Basic coroutine
def simple_coroutine():
    print("Coroutine started")
    x = yield  # Wait for value
    print(f"Received: {x}")

coro = simple_coroutine()
next(coro)  # Prime the coroutine - "Coroutine started"
coro.send(10)  # "Received: 10"

# Coroutine that receives and returns
def averager():
    total = 0
    count = 0
    average = None
    while True:
        value = yield average  # Return average, receive new value
        total += value
        count += 1
        average = total / count

avg = averager()
next(avg)  # Prime it
print(avg.send(10))  # 10.0
print(avg.send(20))  # 15.0
print(avg.send(30))  # 20.0

# Pipeline with coroutines
def producer(consumer):
    for i in range(5):
        consumer.send(i)
    consumer.close()

def filter_even(next_stage):
    try:
        while True:
            value = yield
            if value % 2 == 0:
                next_stage.send(value)
    except GeneratorExit:
        next_stage.close()

def printer():
    try:
        while True:
            value = yield
            print(f"Received: {value}")
    except GeneratorExit:
        print("Printer closed")

# Setup pipeline
p = printer()
next(p)
f = filter_even(p)
next(f)
producer(f)
# Received: 0
# Received: 2
# Received: 4
# Printer closed

# Coroutine decorator
def coroutine(func):
    def wrapper(*args, **kwargs):
        gen = func(*args, **kwargs)
        next(gen)  # Prime automatically
        return gen
    return wrapper

@coroutine
def running_sum():
    total = 0
    while True:
        value = yield total
        total += value

rs = running_sum()  # Already primed!
print(rs.send(10))  # 10
print(rs.send(20))  # 30
print(rs.send(5))   # 35

---

### Q58: How do you implement the Iterator design pattern efficiently?

**Answer:**
The Iterator pattern provides sequential access to elements without exposing internal structure.

```python
# Custom collection with iterator
class BookCollection:
    def __init__(self):
        self.books = []
    
    def add_book(self, book):
        self.books.append(book)
    
    def __iter__(self):
        return BookIterator(self.books)

class BookIterator:
    def __init__(self, books):
        self.books = books
        self.index = 0
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index >= len(self.books):
            raise StopIteration
        book = self.books[self.index]
        self.index += 1
        return book

collection = BookCollection()
collection.add_book("Book 1")
collection.add_book("Book 2")
collection.add_book("Book 3")

for book in collection:
    print(book)

# Reverse iterator
class ReverseIterator:
    def __init__(self, data):
        self.data = data
        self.index = len(data)
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.index == 0:
            raise StopIteration
        self.index -= 1
        return self.data[self.index]

for item in ReverseIterator([1, 2, 3, 4]):
    print(item)  # 4, 3, 2, 1

# Filter iterator
class FilterIterator:
    def __init__(self, iterable, predicate):
        self.iterator = iter(iterable)
        self.predicate = predicate
    
    def __iter__(self):
        return self
    
    def __next__(self):
        while True:
            item = next(self.iterator)
            if self.predicate(item):
                return item

numbers = range(10)
evens = FilterIterator(numbers, lambda x: x % 2 == 0)
print(list(evens))  # [0, 2, 4, 6, 8]

# Tree iterator (in-order traversal)
class TreeNode:
    def __init__(self, value, left=None, right=None):
        self.value = value
        self.left = left
        self.right = right
    
    def __iter__(self):
        return TreeIterator(self)

class TreeIterator:
    def __init__(self, root):
        self.stack = []
        self._push_left(root)
    
    def _push_left(self, node):
        while node:
            self.stack.append(node)
            node = node.left
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if not self.stack:
            raise StopIteration
        
        node = self.stack.pop()
        self._push_left(node.right)
        return node.value

root = TreeNode(2,
    TreeNode(1),
    TreeNode(3)
)

for value in root:
    print(value)  # 1, 2, 3

---

### Q59: How do you use `itertools.tee()` for multiple independent iterators?

**Answer:**
`tee()` splits one iterator into multiple independent iterators for parallel processing.

```python
import itertools

# Basic tee usage
numbers = range(5)
it1, it2 = itertools.tee(numbers, 2)

print(list(it1))  # [0, 1, 2, 3, 4]
print(list(it2))  # [0, 1, 2, 3, 4]

# Process differently
data = range(10)
evens_iter, odds_iter = itertools.tee(data, 2)

evens = [x for x in evens_iter if x % 2 == 0]
odds = [x for x in odds_iter if x % 2 == 1]

print(evens)  # [0, 2, 4, 6, 8]
print(odds)   # [1, 3, 5, 7, 9]

# Pairwise iteration
def pairwise(iterable):
    a, b = itertools.tee(iterable)
    next(b, None)  # Advance b by one
    return zip(a, b)

numbers = [1, 2, 3, 4, 5]
for current, next_val in pairwise(numbers):
    print(f"{current} -> {next_val}")

# Moving average
def moving_average(iterable, n):
    iterators = itertools.tee(iterable, n)
    for i, it in enumerate(iterators):
        for _ in range(i):
            next(it, None)
    
    for values in zip(*iterators):
        yield sum(values) / n

data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
print(list(moving_average(data, 3)))
# [2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

# Warning: tee uses memory to store values
# Only use when iterators won't diverge too much

---

### Q60: What are comprehension alternatives for complex scenarios?

**Answer:**
When comprehensions become unreadable, use explicit loops, generator functions, or functional patterns.

```python
# Complex comprehension - hard to read
# Bad:
result = [
    process(x) if condition1(x) else alternative(x)
    for x in items
    if condition2(x) and condition3(x)
]

# Better: explicit loop
result = []
for x in items:
    if condition2(x) and condition3(x):
        if condition1(x):
            result.append(process(x))
        else:
            result.append(alternative(x))

# Or: generator function
def process_items(items):
    for x in items:
        if condition2(x) and condition3(x):
            if condition1(x):
                yield process(x)
            else:
                yield alternative(x)

result = list(process_items(items))

# Nested loops - can be confusing
# Comprehension:
result = [
    f(x, y)
    for x in range(10)
    if x % 2 == 0
    for y in range(10)
    if y % 3 == 0
]

# Better as explicit loops:
result = []
for x in range(10):
    if x % 2 == 0:
        for y in range(10):
            if y % 3 == 0:
                result.append(f(x, y))

# Use itertools for complex iterations
from itertools import product, combinations, permutations

# Instead of nested comprehensions:
pairs = [(x, y) for x in range(3) for y in range(3)]

# Use product:
pairs = list(product(range(3), repeat=2))

# Complex filtering - use filter + map
from functools import reduce

# Instead of complex comprehension:
result = [x ** 2 for x in range(100) if x % 3 == 0 if x % 5 == 0]

# Use filter + map:
result = list(map(
    lambda x: x ** 2,
    filter(lambda x: x % 3 == 0 and x % 5 == 0, range(100))
))

# Or generator function for clarity:
def filtered_squares():
    for x in range(100):
        if x % 3 == 0 and x % 5 == 0:
            yield x ** 2

result = list(filtered_squares())

---

### Q61: How do you implement efficient caching and memoization?

**Answer:**
Use `functools.lru_cache` for automatic memoization or implement custom caching strategies.

```python
from functools import lru_cache, cache

# Basic memoization with lru_cache
@lru_cache(maxsize=128)
def fibonacci(n):
    if n < 2:
        return n
    return fibonacci(n - 1) + fibonacci(n - 2)

print(fibonacci(100))  # Fast!
print(fibonacci.cache_info())  # Cache statistics

# Unlimited cache (Python 3.9+)
@cache
def expensive_operation(x, y):
    print(f"Computing {x} + {y}")
    return x + y

print(expensive_operation(5, 3))  # "Computing 5 + 3", then 8
print(expensive_operation(5, 3))  # 8 (cached)

# Custom cache decorator
def memo(func):
    cache = {}
    def wrapper(*args):
        if args not in cache:
            cache[args] = func(*args)
        return cache[args]
    return wrapper

@memo
def slow_function(n):
    print(f"Computing for {n}")
    return n ** 2

print(slow_function(5))  # "Computing for 5", then 25
print(slow_function(5))  # 25 (cached)

# Cache with TTL (time-to-live)
import time

def timed_cache(seconds):
    def decorator(func):
        cache = {}
        cache_time = {}
        
        def wrapper(*args):
            now = time.time()
            if args in cache:
                if now - cache_time[args] < seconds:
                    return cache[args]
            
            result = func(*args)
            cache[args] = result
            cache_time[args] = now
            return result
        
        return wrapper
    return decorator

@timed_cache(seconds=5)
def get_data(query):
    print(f"Fetching {query}")
    return f"Data for {query}"

print(get_data("test"))  # "Fetching test"
print(get_data("test"))  # Cached
time.sleep(6)
print(get_data("test"))  # "Fetching test" again (cache expired)

# LRU cache with size limit
from collections import OrderedDict

class LRUCache:
    def __init__(self, capacity):
        self.cache = OrderedDict()
        self.capacity = capacity
    
    def get(self, key):
        if key not in self.cache:
            return None
        self.cache.move_to_end(key)  # Mark as recently used
        return self.cache[key]
    
    def put(self, key, value):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            self.cache.popitem(last=False)  # Remove oldest

cache = LRUCache(3)
cache.put("a", 1)
cache.put("b", 2)
cache.put("c", 3)
cache.put("d", 4)  # "a" evicted
print(cache.get("a"))  # None

# Property-level caching
from functools import cached_property

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    @cached_property
    def mean(self):
        print("Computing mean...")
        return sum(self.data) / len(self.data)

processor = DataProcessor([1, 2, 3, 4, 5])
print(processor.mean)  # "Computing mean...", then 3.0
print(processor.mean)  # 3.0 (cached)

# Conditional caching
def conditional_cache(condition_func):
    def decorator(func):
        cache = {}
        def wrapper(*args):
            if condition_func(*args):
                if args not in cache:
                    cache[args] = func(*args)
                return cache[args]
            return func(*args)
        return wrapper
    return decorator

@conditional_cache(lambda x: x > 10)
def process(x):
    print(f"Processing {x}")
    return x ** 2

print(process(5))   # "Processing 5" (not cached)
print(process(5))   # "Processing 5" again
print(process(15))  # "Processing 15" (cached)
print(process(15))  # Returned from cache

---

### Q62: How do you optimize memory usage with `__slots__`?

**Answer:**
`__slots__` restricts attributes and reduces memory overhead by preventing `__dict__` creation.

```python
# Without slots - uses __dict__
class PersonNormal:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# With slots - more memory efficient
class PersonSlots:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Memory comparison
import sys

normal = PersonNormal("Alice", 30)
slotted = PersonSlots("Bob", 25)

print(f"Normal: {sys.getsizeof(normal) + sys.getsizeof(normal.__dict__)} bytes")
print(f"Slotted: {sys.getsizeof(slotted)} bytes")

# Slots prevent dynamic attribute addition
# normal.email = "alice@example.com"  # Works
# slotted.email = "bob@example.com"   # AttributeError!

# Slots with inheritance
class Person:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

class Employee(Person):
    __slots__ = ['employee_id']  # Add more slots
    
    def __init__(self, name, age, employee_id):
        super().__init__(name, age)
        self.employee_id = employee_id

emp = Employee("Charlie", 35, "E123")

# Slots with default values (Python 3.10+)
class Point:
    __slots__ = ('x', 'y')
    
    def __init__(self, x=0, y=0):
        self.x = x
        self.y = y

# When to use slots:
# ✓ Creating millions of instances
# ✓ Memory-constrained environments
# ✓ Data classes with fixed attributes
# ✗ Need dynamic attributes
# ✗ Multiple inheritance scenarios
# ✗ When flexibility > performance

# Real-world example: Point class
class Point3D:
    __slots__ = ['x', 'y', 'z']
    
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    def distance(self):
        return (self.x**2 + self.y**2 + self.z**2)**0.5

# Create million points - significant memory savings
points = [Point3D(i, i+1, i+2) for i in range(1000000)]

---

### Q63: What are memory views and how do they work?

**Answer:**
Memory views provide zero-copy access to array data, efficient for large binary data manipulation.

```python
# Without memoryview - creates copies
data = bytearray(b'Hello World')
slice1 = data[0:5]  # Creates copy
slice1[0] = ord('h')
print(data)  # b'Hello World' - original unchanged

# With memoryview - no copy
data = bytearray(b'Hello World')
view = memoryview(data)
slice_view = view[0:5]  # No copy!
slice_view[0] = ord('h')
print(data)  # b'hello World' - original changed!

# Performance with large data
import array

# Create large array
numbers = array.array('i', range(1000000))

# Memory efficient slicing
view = memoryview(numbers)
chunk = view[100:200]  # No copy created!

# Modify through memoryview
view[0] = 999
print(numbers[0])  # 999 - original modified

# Multi-dimensional data
import numpy as np  # If numpy available

# 2D array view
# arr = np.array([[1, 2, 3], [4, 5, 6]])
# view = memoryview(arr)

# Working with different formats
data = bytearray(8)  # 8 bytes
view = memoryview(data)

# Cast to different type
int_view = view.cast('i')  # View as integers
int_view[0] = 12345

print(data.hex())  # Shows bytes

# Reading binary file efficiently
def read_large_binary(filename):
    with open(filename, 'rb') as f:
        data = f.read()
        view = memoryview(data)
        # Process without copying
        for i in range(0, len(view), 1024):
            chunk = view[i:i+1024]
            process_chunk(chunk)

# Benefits:
# - No memory copies
# - Efficient slicing
# - Direct buffer access
# - Good for binary protocols

---

### Q64: How do you implement efficient data structures?

**Answer:**
Choose the right data structure and optimize access patterns for your use case.

```python
from collections import deque, defaultdict, Counter, OrderedDict
import heapq

# deque - O(1) append/pop from both ends
queue = deque()
queue.append(1)  # O(1) - right
queue.appendleft(0)  # O(1) - left
queue.pop()  # O(1) - right
queue.popleft()  # O(1) - left

# List is O(n) for left operations!
regular_list = []
regular_list.append(1)  # O(1)
regular_list.insert(0, 0)  # O(n) - slow!

# Use deque for queues
from collections import deque

def bfs(graph, start):
    visited = set()
    queue = deque([start])
    
    while queue:
        node = queue.popleft()  # O(1)
        if node not in visited:
            visited.add(node)
            queue.extend(graph[node])

# defaultdict - automatic default values
word_count = defaultdict(int)
for word in ["hello", "world", "hello"]:
    word_count[word] += 1  # No KeyError!

print(dict(word_count))  # {'hello': 2, 'world': 1}

# Grouping with defaultdict
from collections import defaultdict

students = [
    ('Alice', 'Math'),
    ('Bob', 'Science'),
    ('Charlie', 'Math')
]

by_subject = defaultdict(list)
for name, subject in students:
    by_subject[subject].append(name)

# Counter - counting made easy
from collections import Counter

text = "hello world hello python"
word_count = Counter(text.split())
print(word_count)  # Counter({'hello': 2, 'world': 1, 'python': 1})

# Most common
print(word_count.most_common(2))  # [('hello', 2), ('world', 1)]

# Heap (priority queue) - O(log n) operations
heap = []
heapq.heappush(heap, (3, 'task3'))  # (priority, task)
heapq.heappush(heap, (1, 'task1'))
heapq.heappush(heap, (2, 'task2'))

while heap:
    priority, task = heapq.heappop(heap)
    print(f"{task}: {priority}")
# task1: 1
# task2: 2
# task3: 3

# OrderedDict - remembers insertion order (dict is ordered in 3.7+)
from collections import OrderedDict

cache = OrderedDict()
cache['a'] = 1
cache['b'] = 2
cache['c'] = 3

# LRU cache behavior
cache.move_to_end('a')  # Move 'a' to end
cache.popitem(last=False)  # Remove oldest ('b')

# Set for O(1) lookups
large_list = list(range(1000000))
large_set = set(range(1000000))

# Slow
# 999999 in large_list  # O(n)

# Fast
999999 in large_set  # O(1)

# Bisect for sorted lists
import bisect

sorted_list = [1, 3, 5, 7, 9]
# Insert maintaining order
bisect.insort(sorted_list, 4)  # [1, 3, 4, 5, 7, 9]

# Find insertion point
pos = bisect.bisect_left(sorted_list, 5)  # 3

# Named tuples - memory efficient records
from collections import namedtuple

Point = namedtuple('Point', ['x', 'y'])
p = Point(10, 20)
print(p.x, p.y)  # 10 20
print(p[0], p[1])  # Also works

# Performance comparison
import timeit

# List append vs deque append
list_time = timeit.timeit('l.append(1); l.pop(0)', 
                          setup='l=list(range(1000))', number=10000)
deque_time = timeit.timeit('d.append(1); d.popleft()', 
                           setup='from collections import deque; d=deque(range(1000))', 
                           number=10000)

print(f"List: {list_time:.4f}s")
print(f"Deque: {deque_time:.4f}s")
# Deque is much faster!

---

### Q65: How do you profile and optimize memory usage?

**Answer:**
Use memory profilers to identify bottlenecks and optimize data structures and algorithms.

```python
# Basic memory tracking
import sys

data = list(range(1000))
print(f"List size: {sys.getsizeof(data)} bytes")

# More accurate memory usage
def get_size(obj):
    size = sys.getsizeof(obj)
    if isinstance(obj, dict):
        size += sum(get_size(k) + get_size(v) for k, v in obj.items())
    elif isinstance(obj, (list, tuple, set)):
        size += sum(get_size(item) for item in obj)
    return size

nested_data = {'a': [1, 2, 3], 'b': [4, 5, 6]}
print(f"Total size: {get_size(nested_data)} bytes")

# Memory profiling with memory_profiler
# pip install memory-profiler
# @profile  # Uncomment when using memory_profiler
def memory_intensive():
    # Large list
    data = [i for i in range(1000000)]
    # Process
    result = [x ** 2 for x in data]
    return result

# Run with: python -m memory_profiler script.py

# Generator alternative - uses constant memory
def memory_efficient():
    data = range(1000000)  # Generator
    result = (x ** 2 for x in data)  # Generator
    return result

# Tracemalloc - built-in memory tracking
import tracemalloc

tracemalloc.start()

# Code to profile
data = [i for i in range(100000)]

current, peak = tracemalloc.get_traced_memory()
print(f"Current: {current / 1024:.2f} KB")
print(f"Peak: {peak / 1024:.2f} KB")

tracemalloc.stop()

# Find memory leaks
def snapshot_comparison():
    import tracemalloc
    
    tracemalloc.start()
    snapshot1 = tracemalloc.take_snapshot()
    
    # Code that might leak
    leaked = []
    for i in range(10000):
        leaked.append([i] * 100)
    
    snapshot2 = tracemalloc.take_snapshot()
    
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    for stat in top_stats[:5]:
        print(stat)

# Object reference counting
import sys

data = [1, 2, 3]
print(sys.getrefcount(data))  # 2 (one from variable, one from argument)

# Weak references - don't prevent garbage collection
import weakref

class BigObject:
    pass

obj = BigObject()
weak_ref = weakref.ref(obj)

print(weak_ref())  # <BigObject object>
del obj
print(weak_ref())  # None - object was garbage collected

# Memory optimization tips:
# 1. Use generators instead of lists when possible
# 2. Use __slots__ for classes with many instances
# 3. Use array.array for homogeneous numeric data
# 4. Use sets for membership testing
# 5. Delete large objects explicitly when done
# 6. Use itertools for efficient iteration
# 7. Profile before optimizing!

---

## Section 5: Concurrency, Parallelism & Async Programming (Q66-80)

### Q66: What's the difference between concurrency and parallelism?

**Answer:**
Concurrency is about dealing with multiple tasks, parallelism is about doing multiple tasks simultaneously.

```python
# Concurrency: Multiple tasks in progress (not necessarily simultaneous)
# - Threading: Good for I/O-bound tasks
# - AsyncIO: Best for I/O-bound tasks
#
# Parallelism: Multiple tasks executed simultaneously
# - Multiprocessing: Good for CPU-bound tasks

# Example: Concurrent (one cook, multiple dishes)
import threading
import time

def cook_dish(dish_name):
    print(f"Start cooking {dish_name}")
    time.sleep(2)  # Simulate cooking
    print(f"Finished {dish_name}")

threads = []
for dish in ["pasta", "salad", "soup"]:
    thread = threading.Thread(target=cook_dish, args=(dish,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

# Example: Parallel (multiple cooks, each cooking own dish)
import multiprocessing

def process_data(chunk):
    return sum(x ** 2 for x in chunk)

if __name__ == '__main__':
    data = range(1000000)
    chunks = [range(i, i+250000) for i in range(0, 1000000, 250000)]
    
    with multiprocessing.Pool(4) as pool:
        results = pool.map(process_data, chunks)
    
    print(sum(results))

# Async: Concurrent without threads
import asyncio

async def fetch_data(url):
    print(f"Fetching {url}")
    await asyncio.sleep(1)  # Simulate network request
    return f"Data from {url}"

async def main():
    urls = ["url1", "url2", "url3"]
    tasks = [fetch_data(url) for url in urls]
    results = await asyncio.gather(*tasks)
    print(results)

# asyncio.run(main())

# When to use what:
# - Threading: I/O-bound (file, network), light concurrency
# - AsyncIO: I/O-bound, many concurrent operations
# - Multiprocessing: CPU-bound, true parallelism needed

---

### Q67: How do you use threading in Python effectively?

**Answer:**
Threading enables concurrent execution but is limited by the GIL for CPU-bound tasks.

```python
import threading
import time

# Basic thread
def worker(name):
    print(f"Thread {name} starting")
    time.sleep(2)
    print(f"Thread {name} done")

thread = threading.Thread(target=worker, args=("A",))
thread.start()
thread.join()  # Wait for completion

# Multiple threads
threads = []
for i in range(5):
    thread = threading.Thread(target=worker, args=(i,))
    threads.append(thread)
    thread.start()

for thread in threads:
    thread.join()

# Thread-safe counter with Lock
counter = 0
lock = threading.Lock()

def increment():
    global counter
    for _ in range(100000):
        with lock:  # Thread-safe
            counter += 1

threads = [threading.Thread(target=increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)  # 1000000 (correct with lock)

# Without lock - race condition!
# counter would be less than 1000000

# Thread pool for managing multiple threads
from concurrent.futures import ThreadPoolExecutor

def task(n):
    time.sleep(1)
    return n * 2

with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(task, i) for i in range(10)]
    results = [f.result() for f in futures]

print(results)

# Map with thread pool
with ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(task, range(10)))

# Daemon threads - stop when main thread stops
def daemon_worker():
    while True:
        print("Daemon working...")
        time.sleep(1)

daemon = threading.Thread(target=daemon_worker, daemon=True)
daemon.start()
time.sleep(3)  # Daemon stops when program exits

# Thread-local storage
thread_local = threading.local()

def worker():
    thread_local.value = threading.current_thread().name
    print(f"Thread {thread_local.value}")

threads = [threading.Thread(target=worker) for _ in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Semaphore - limit concurrent access
semaphore = threading.Semaphore(3)  # Max 3 concurrent

def limited_access(n):
    with semaphore:
        print(f"Thread {n} accessing")
        time.sleep(1)
        print(f"Thread {n} done")

threads = [threading.Thread(target=limited_access, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Event for thread synchronization
event = threading.Event()

def waiter():
    print("Waiting for event...")
    event.wait()  # Block until set
    print("Event received!")

def setter():
    time.sleep(2)
    print("Setting event")
    event.set()

t1 = threading.Thread(target=waiter)
t2 = threading.Thread(target=setter)
t1.start()
t2.start()
t1.join()
t2.join()

---

### Q68: What is the Global Interpreter Lock (GIL) and how does it affect Python?

**Answer:**
The GIL is a mutex that prevents multiple threads from executing Python bytecode simultaneously.

```python
import threading
import time

# GIL Impact on CPU-bound tasks
def cpu_bound():
    total = 0
    for i in range(10_000_000):
        total += i
    return total

# Single thread
start = time.time()
result1 = cpu_bound()
result2 = cpu_bound()
print(f"Sequential: {time.time() - start:.2f}s")

# Multiple threads - NOT faster due to GIL!
start = time.time()
thread1 = threading.Thread(target=cpu_bound)
thread2 = threading.Thread(target=cpu_bound)
thread1.start()
thread2.start()
thread1.join()
thread2.join()
print(f"Threading: {time.time() - start:.2f}s")  # Same or slower!

# Multiprocessing bypasses GIL
from multiprocessing import Process

start = time.time()
proc1 = Process(target=cpu_bound)
proc2 = Process(target=cpu_bound)
proc1.start()
proc2.start()
proc1.join()
proc2.join()
print(f"Multiprocessing: {time.time() - start:.2f}s")  # Faster!

# Threading is good for I/O-bound tasks
import requests  # Example

def io_bound(url):
    response = requests.get(url)
    return len(response.content)

# Threading works well here because waiting for I/O releases GIL
urls = ["https://example.com"] * 10

start = time.time()
with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(io_bound, urls))
print(f"Threading I/O: {time.time() - start:.2f}s")  # Fast!

# GIL workarounds:
# 1. Multiprocessing for CPU-bound
# 2. AsyncIO for I/O-bound
# 3. C extensions (NumPy, etc) release GIL
# 4. Use different Python implementation (Jython, IronPython)

---

### Q69: How do you use multiprocessing for CPU-bound tasks?

**Answer:**
Multiprocessing creates separate Python processes, each with its own GIL, enabling true parallelism.

```python
from multiprocessing import Process, Pool, Queue, Manager, Value, Array
import time

# Basic process
def worker(name):
    print(f"Process {name} starting")
    time.sleep(1)
    print(f"Process {name} done")

if __name__ == '__main__':
    process = Process(target=worker, args=("A",))
    process.start()
    process.join()

# Process Pool - easier management
def square(x):
    return x ** 2

if __name__ == '__main__':
    with Pool(processes=4) as pool:
        results = pool.map(square, range(10))
    print(results)

# Parallel processing of large dataset
def process_chunk(chunk):
    return sum(x ** 2 for x in chunk)

if __name__ == '__main__':
    data = range(1_000_000)
    chunk_size = 250_000
    chunks = [range(i, i+chunk_size) for i in range(0, 1_000_000, chunk_size)]
    
    with Pool(4) as pool:
        results = pool.map(process_chunk, chunks)
    
    total = sum(results)
    print(total)

# Shared memory between processes
from multiprocessing import Value, Array

def increment(shared_value, shared_array):
    shared_value.value += 1
    for i in range(len(shared_array)):
        shared_array[i] += 1

if __name__ == '__main__':
    shared_num = Value('i', 0)  # Shared integer
    shared_arr = Array('i', [0, 0, 0])  # Shared array
    
    processes = [Process(target=increment, args=(shared_num, shared_arr)) for _ in range(5)]
    for p in processes:
        p.start()
    for p in processes:
        p.join()
    
    print(shared_num.value)  # 5
    print(list(shared_arr))  # [5, 5, 5]

# Queue for inter-process communication
from multiprocessing import Queue

def producer(queue):
    for i in range(5):
        queue.put(i)
    queue.put(None)  # Signal done

def consumer(queue):
    while True:
        item = queue.get()
        if item is None:
            break
        print(f"Consumed: {item}")

if __name__ == '__main__':
    q = Queue()
    prod = Process(target=producer, args=(q,))
    cons = Process(target=consumer, args=(q,))
    
    prod.start()
    cons.start()
    prod.join()
    cons.join()

# Manager for complex shared objects
from multiprocessing import Manager

def update_dict(shared_dict, key, value):
    shared_dict[key] = value

if __name__ == '__main__':
    with Manager() as manager:
        shared_dict = manager.dict()
        processes = []
        
        for i in range(5):
            p = Process(target=update_dict, args=(shared_dict, f'key{i}', i))
            processes.append(p)
            p.start()
        
        for p in processes:
            p.join()
        
        print(dict(shared_dict))

# Process Pool with error handling
def risky_operation(x):
    if x == 5:
        raise ValueError("Bad value!")
    return x ** 2

if __name__ == '__main__':
    with Pool(4) as pool:
        try:
            results = pool.map(risky_operation, range(10))
        except ValueError as e:
            print(f"Error: {e}")

---

### Q70: How do you use asyncio for asynchronous programming?

**Answer:**
AsyncIO enables concurrent I/O operations using coroutines and an event loop.

```python
import asyncio

# Basic async function
async def say_hello():
    print("Hello")
    await asyncio.sleep(1)  # Non-blocking sleep
    print("World")

# Run async function
asyncio.run(say_hello())

# Multiple concurrent tasks
async def fetch_data(name):
    print(f"Fetching {name}")
    await asyncio.sleep(2)  # Simulate API call
    return f"Data from {name}"

async def main():
    # Run concurrently
    results = await asyncio.gather(
        fetch_data("API1"),
        fetch_data("API2"),
        fetch_data("API3")
    )
    print(results)

asyncio.run(main())

# Create tasks
async def main_with_tasks():
    task1 = asyncio.create_task(fetch_data("API1"))
    task2 = asyncio.create_task(fetch_data("API2"))
    
    # Do other work while tasks run
    print("Tasks started")
    
    # Wait for completion
    result1 = await task1
    result2 = await task2
    print(result1, result2)

asyncio.run(main_with_tasks())

# Async HTTP requests
import aiohttp

async def fetch_url(session, url):
    async with session.get(url) as response:
        return await response.text()

async def fetch_all():
    urls = [
        "https://example.com",
        "https://example.org",
        "https://example.net"
    ]
    
    async with aiohttp.ClientSession() as session:
        tasks = [fetch_url(session, url) for url in urls]
        results = await asyncio.gather(*tasks)
    return results

# asyncio.run(fetch_all())

# Async context manager
class AsyncResource:
    async def __aenter__(self):
        print("Acquiring resource")
        await asyncio.sleep(1)
        return self
    
    async def __aexit__(self, *args):
        print("Releasing resource")
        await asyncio.sleep(1)

async def use_resource():
    async with AsyncResource() as resource:
        print("Using resource")

asyncio.run(use_resource())

# Async iterator
class AsyncCounter:
    def __init__(self, stop):
        self.current = 0
        self.stop = stop
    
    def __aiter__(self):
        return self
    
    async def __anext__(self):
        if self.current >= self.stop:
            raise StopAsyncIteration
        await asyncio.sleep(0.1)
        self.current += 1
        return self.current

async def count():
    async for number in AsyncCounter(5):
        print(number)

asyncio.run(count())

# Timeout handling
async def slow_operation():
    await asyncio.sleep(10)
    return "Done"

async def with_timeout():
    try:
        result = await asyncio.wait_for(slow_operation(), timeout=2)
    except asyncio.TimeoutError:
        print("Operation timed out")

asyncio.run(with_timeout())

# Running sync code in async
import time

async def run_blocking():
    loop = asyncio.get_event_loop()
    # Run CPU-bound in thread pool
    result = await loop.run_in_executor(None, time.sleep, 2)
    return "Done"

---

### Q71: How do you handle race conditions and thread safety?

**Answer:**
Use locks, semaphores, and thread-safe data structures to prevent race conditions.

```python
import threading

# Race condition example - UNSAFE
counter = 0

def unsafe_increment():
    global counter
    for _ in range(100000):
        counter += 1  # NOT atomic!

threads = [threading.Thread(target=unsafe_increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)  # Less than 1000000 due to race condition!

# Thread-safe with Lock
counter = 0
lock = threading.Lock()

def safe_increment():
    global counter
    for _ in range(100000):
        with lock:
            counter += 1

threads = [threading.Thread(target=safe_increment) for _ in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()

print(counter)  # 1000000 - correct!

# RLock - Reentrant Lock (can be acquired multiple times by same thread)
class BankAccount:
    def __init__(self):
        self.balance = 0
        self.lock = threading.RLock()
    
    def deposit(self, amount):
        with self.lock:
            self.balance += amount
    
    def withdraw(self, amount):
        with self.lock:
            if self.balance >= amount:
                self.balance -= amount
                return True
            return False
    
    def transfer(self, other, amount):
        with self.lock:  # Can acquire again
            if self.withdraw(amount):
                other.deposit(amount)

# Thread-safe queue
from queue import Queue, Empty

queue = Queue()

def producer():
    for i in range(5):
        queue.put(i)
        threading.Event().wait(0.1)

def consumer():
    while True:
        try:
            item = queue.get(timeout=1)
            print(f"Consumed: {item}")
            queue.task_done()
        except Empty:
            break

prod = threading.Thread(target=producer)
cons = threading.Thread(target=consumer)
prod.start()
cons.start()
prod.join()
cons.join()

# Condition variable for complex synchronization
condition = threading.Condition()
items = []

def consumer_cv():
    with condition:
        while not items:
            condition.wait()  # Wait for notification
        item = items.pop(0)
        print(f"Consumed: {item}")

def producer_cv():
    with condition:
        items.append(1)
        condition.notify()  # Wake up consumer

# Atomic operations with threading
from threading import Lock

class AtomicCounter:
    def __init__(self):
        self._value = 0
        self._lock = Lock()
    
    def increment(self):
        with self._lock:
            self._value += 1
            return self._value
    
    def get(self):
        with self._lock:
            return self._value

---

### Q72: How do you use concurrent.futures for high-level concurrency?

**Answer:**
`concurrent.futures` provides a high-level interface for both threading and multiprocessing.

```python
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import time

# Thread pool
def task(n):
    time.sleep(1)
    return n * 2

with ThreadPoolExecutor(max_workers=5) as executor:
    # submit for individual tasks
    future = executor.submit(task, 5)
    result = future.result()
    print(result)  # 10
    
    # map for multiple tasks
    results = executor.map(task, range(10))
    print(list(results))

# Process pool for CPU-bound
def cpu_intensive(n):
    return sum(range(n))

with ProcessPoolExecutor(max_workers=4) as executor:
    results = executor.map(cpu_intensive, [1000000, 2000000, 3000000])
    print(list(results))

# as_completed - process results as they finish
with ThreadPoolExecutor(max_workers=5) as executor:
    futures = [executor.submit(task, i) for i in range(10)]
    
    for future in as_completed(futures):
        result = future.result()
        print(f"Got result: {result}")

# Error handling
def risky_task(n):
    if n == 5:
        raise ValueError("Bad value!")
    return n * 2

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [executor.submit(risky_task, i) for i in range(10)]
    
    for future in as_completed(futures):
        try:
            result = future.result()
            print(result)
        except ValueError as e:
            print(f"Error: {e}")

# Timeout handling
with ThreadPoolExecutor(max_workers=2) as executor:
    future = executor.submit(time.sleep, 10)
    try:
        result = future.result(timeout=2)
    except TimeoutError:
        print("Task timed out")

# Canceling futures
with ThreadPoolExecutor(max_workers=2) as executor:
    future = executor.submit(time.sleep, 10)
    time.sleep(1)
    cancelled = future.cancel()  # True if cancelled
    print(f"Cancelled: {cancelled}")

# Real-world example: web scraping
import requests

def fetch_url(url):
    response = requests.get(url)
    return (url, len(response.content))

urls = [f"https://example.com/page{i}" for i in range(10)]

with ThreadPoolExecutor(max_workers=10) as executor:
    results = list(executor.map(fetch_url, urls))
    for url, size in results:
        print(f"{url}: {size} bytes")

---

### Q73: How do you implement producer-consumer patterns?

**Answer:**
Use queues for thread-safe producer-consumer communication.

```python
import threading
import queue
import time

# Basic producer-consumer with Queue
def producer(q, items):
    for item in items:
        print(f"Producing {item}")
        q.put(item)
        time.sleep(0.1)
    q.put(None)  # Sentinel value

def consumer(q):
    while True:
        item = q.get()
        if item is None:
            break
        print(f"Consuming {item}")
        time.sleep(0.2)
        q.task_done()

q = queue.Queue()
prod = threading.Thread(target=producer, args=(q, range(5)))
cons = threading.Thread(target=consumer, args=(q,))

prod.start()
cons.start()
prod.join()
cons.join()

# Multiple producers and consumers
def multi_producer(q, name, items):
    for item in items:
        q.put(f"{name}-{item}")
    print(f"Producer {name} done")

def multi_consumer(q, name):
    while True:
        try:
            item = q.get(timeout=1)
            print(f"Consumer {name} got {item}")
            q.task_done()
        except queue.Empty:
            break

q = queue.Queue()

# Start producers
producers = [
    threading.Thread(target=multi_producer, args=(q, f"P{i}", range(3)))
    for i in range(2)
]

# Start consumers
consumers = [
    threading.Thread(target=multi_consumer, args=(q, f"C{i}"))
    for i in range(3)
]

for p in producers:
    p.start()
for c in consumers:
    c.start()

for p in producers:
    p.join()
q.join()  # Wait for all items to be processed

# Priority queue
priority_q = queue.PriorityQueue()

def priority_producer(q):
    q.put((1, "Low priority"))
    q.put((5, "High priority"))
    q.put((3, "Medium priority"))

def priority_consumer(q):
    while not q.empty():
        priority, item = q.get()
        print(f"Processing: {item} (priority: {priority})")

priority_producer(priority_q)
priority_consumer(priority_q)

# Bounded queue (max size)
bounded_q = queue.Queue(maxsize=3)

def bounded_producer(q):
    for i in range(10):
        q.put(i)  # Blocks if queue is full
        print(f"Produced {i}")

def bounded_consumer(q):
    while True:
        item = q.get()
        print(f"Consumed {item}")
        time.sleep(1)  # Slow consumer
        q.task_done()

---

### Q74: How do you use asyncio.Queue for async producer-consumer?

**Answer:**
AsyncIO provides async queues for coroutine-based producer-consumer patterns.

```python
import asyncio

async def producer(queue, n):
    for i in range(n):
        await asyncio.sleep(0.1)
        await queue.put(i)
        print(f"Produced {i}")
    await queue.put(None)  # Signal done

async def consumer(queue):
    while True:
        item = await queue.get()
        if item is None:
            break
        await asyncio.sleep(0.2)
        print(f"Consumed {item}")

async def main():
    queue = asyncio.Queue()
    
    await asyncio.gather(
        producer(queue, 5),
        consumer(queue)
    )

asyncio.run(main())

# Multiple async producers/consumers
async def multi_producer(queue, name, n):
    for i in range(n):
        await queue.put(f"{name}-{i}")
        await asyncio.sleep(0.1)

async def multi_consumer(queue, name):
    while True:
        try:
            item = await asyncio.wait_for(queue.get(), timeout=2)
            print(f"{name} consumed {item}")
        except asyncio.TimeoutError:
            break

async def main_multi():
    queue = asyncio.Queue()
    
    await asyncio.gather(
        multi_producer(queue, "P1", 3),
        multi_producer(queue, "P2", 3),
        multi_consumer(queue, "C1"),
        multi_consumer(queue, "C2")
    )

asyncio.run(main_multi())

---

### Q75: How do you handle deadlocks in Python?

**Answer:**
Prevent deadlocks through lock ordering, timeouts, and proper resource management.

```python
import threading
import time

# Deadlock example - TWO locks acquired in different order
lock1 = threading.Lock()
lock2 = threading.Lock()

def task1():
    with lock1:
        print("Task 1 has lock 1")
        time.sleep(0.1)
        with lock2:
            print("Task 1 has lock 2")

def task2():
    with lock2:  # Different order - DEADLOCK!
        print("Task 2 has lock 2")
        time.sleep(0.1)
        with lock1:
            print("Task 2 has lock 1")

# This will deadlock:
# t1 = threading.Thread(target=task1)
# t2 = threading.Thread(target=task2)
# t1.start()
# t2.start()

# Solution 1: Consistent lock ordering
def safe_task1():
    with lock1:
        with lock2:
            print("Safe task 1")

def safe_task2():
    with lock1:  # Same order
        with lock2:
            print("Safe task 2")

# Solution 2: Try-lock with timeout
def try_task():
    if lock1.acquire(timeout=1):
        try:
            if lock2.acquire(timeout=1):
                try:
                    print("Got both locks")
                finally:
                    lock2.release()
        finally:
            lock1.release()
    else:
        print("Couldn't acquire locks")

# Solution 3: Context manager with timeout
from contextlib import contextmanager

@contextmanager
def acquire_timeout(lock, timeout):
    result = lock.acquire(timeout=timeout)
    try:
        yield result
    finally:
        if result:
            lock.release()

def safe_with_timeout():
    with acquire_timeout(lock1, 1) as got_lock1:
        if got_lock1:
            with acquire_timeout(lock2, 1) as got_lock2:
                if got_lock2:
                    print("Success")

# Deadlock detection (simple)
import threading

class DeadlockDetector:
    def __init__(self):
        self.locks = {}
        self.lock = threading.Lock()
    
    def acquire(self, thread_id, lock_id):
        with self.lock:
            if thread_id not in self.locks:
                self.locks[thread_id] = []
            self.locks[thread_id].append(lock_id)
            
            # Check for circular wait
            if self.has_cycle():
                print("DEADLOCK DETECTED!")
                return False
            return True
    
    def has_cycle(self):
        # Simplified cycle detection
        # In real implementation, use proper graph algorithm
        return False

---

### Q76: How do you implement timeouts in concurrent code?

**Answer:**
Use timeout parameters and context managers to prevent indefinite blocking.

```python
import threading
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

# Thread join with timeout
def long_running():
    time.sleep(10)

thread = threading.Thread(target=long_running)
thread.start()
thread.join(timeout=2)  # Wait max 2 seconds

if thread.is_alive():
    print("Thread still running after timeout")

# Lock with timeout
lock = threading.Lock()

def try_lock():
    if lock.acquire(timeout=2):
        try:
            print("Got lock")
        finally:
            lock.release()
    else:
        print("Lock timeout")

# Queue get with timeout
from queue import Queue, Empty

q = Queue()

try:
    item = q.get(timeout=1)
except Empty:
    print("Queue get timed out")

# Future with timeout
with ThreadPoolExecutor() as executor:
    future = executor.submit(time.sleep, 10)
    try:
        result = future.result(timeout=2)
    except TimeoutError:
        print("Future timed out")
        future.cancel()

# Asyncio timeout
import asyncio

async def slow_operation():
    await asyncio.sleep(10)

async def with_timeout():
    try:
        await asyncio.wait_for(slow_operation(), timeout=2)
    except asyncio.TimeoutError:
        print("Async operation timed out")

asyncio.run(with_timeout())

# Context manager for timeout
import signal
from contextlib import contextmanager

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutError("Timed out!")
    
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

# Use:
# try:
#     with time_limit(5):
#         long_running_task()
# except TimeoutError:
#     print("Task timed out")

---

### Q77: How do you use semaphores and barriers?

**Answer:**
Semaphores limit concurrent access, barriers synchronize multiple threads.

```python
import threading
import time

# Semaphore - limit concurrent access
semaphore = threading.Semaphore(3)  # Max 3 concurrent

def limited_resource(n):
    with semaphore:
        print(f"Thread {n} accessing")
        time.sleep(2)
        print(f"Thread {n} done")

threads = [threading.Thread(target=limited_resource, args=(i,)) for i in range(10)]
for t in threads:
    t.start()
for t in threads:
    t.join()
# Only 3 threads run simultaneously

# BoundedSemaphore - prevents release() above initial value
bounded = threading.BoundedSemaphore(2)

# Barrier - wait for all threads
barrier = threading.Barrier(3)  # Wait for 3 threads

def worker(n):
    print(f"Thread {n} starting")
    time.sleep(n)
    print(f"Thread {n} waiting at barrier")
    barrier.wait()  # All threads wait here
    print(f"Thread {n} passed barrier")

threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Real example: rate limiting
class RateLimiter:
    def __init__(self, max_calls, time_window):
        self.semaphore = threading.Semaphore(max_calls)
        self.time_window = time_window
    
    def __call__(self, func):
        def wrapper(*args, **kwargs):
            self.semaphore.acquire()
            try:
                result = func(*args, **kwargs)
                # Release after time window
                threading.Timer(self.time_window, self.semaphore.release).start()
                return result
            except:
                self.semaphore.release()
                raise
        return wrapper

@RateLimiter(max_calls=5, time_window=1)
def api_call():
    print("API call made")

---

### Q78: How do you implement async context managers?

**Answer:**
Async context managers use `async with` for async setup/teardown.

```python
import asyncio

# Basic async context manager
class AsyncDatabase:
    async def __aenter__(self):
        print("Connecting to database")
        await asyncio.sleep(1)
        self.connection = "DB Connection"
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        print("Closing database")
        await asyncio.sleep(1)
        return False

async def use_db():
    async with AsyncDatabase() as db:
        print(f"Using {db.connection}")

asyncio.run(use_db())

# Async context manager with asynccontextmanager
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_timer(label):
    import time
    start = time.time()
    try:
        yield
    finally:
        end = time.time()
        print(f"{label}: {end - start:.2f}s")

async def timed_operation():
    async with async_timer("Operation"):
        await asyncio.sleep(2)
        print("Working...")

asyncio.run(timed_operation())

# File handling async
@asynccontextmanager
async def async_open(filename, mode='r'):
    # In real code, use aiofiles library
    file = open(filename, mode)
    try:
        yield file
    finally:
        file.close()

# HTTP session management
import aiohttp

@asynccontextmanager
async def http_session():
    session = aiohttp.ClientSession()
    try:
        yield session
    finally:
        await session.close()

async def fetch_data():
    async with http_session() as session:
        async with session.get('https://example.com') as response:
            return await response.text()

---

### Q79: How do you handle signals in async code?

**Answer:**
Use asyncio's signal handling for graceful shutdown and interruption.

```python
import asyncio
import signal

# Basic signal handling
async def main():
    loop = asyncio.get_running_loop()
    
    def handle_signal(sig):
        print(f"Received signal {sig}")
        loop.stop()
    
    for sig in (signal.SIGTERM, signal.SIGINT):
        loop.add_signal_handler(sig, lambda s=sig: handle_signal(s))
    
    try:
        while True:
            await asyncio.sleep(1)
            print("Working...")
    except KeyboardInterrupt:
        print("Interrupted")

# asyncio.run(main())

# Graceful shutdown
class Application:
    def __init__(self):
        self.running = True
        self.tasks = []
    
    async def worker(self, name):
        while self.running:
            print(f"{name} working")
            await asyncio.sleep(1)
        print(f"{name} shutting down")
    
    def shutdown(self):
        print("Shutting down...")
        self.running = False
    
    async def run(self):
        loop = asyncio.get_running_loop()
        
        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, self.shutdown)
        
        self.tasks = [
            asyncio.create_task(self.worker(f"Worker-{i}"))
            for i in range(3)
        ]
        
        await asyncio.gather(*self.tasks)

# app = Application()
# asyncio.run(app.run())

---

### Q80: How do you debug concurrent Python code?

**Answer:**
Use logging, threading utilities, and specialized tools to debug concurrent issues.

```python
import threading
import logging
import time

# Configure logging for threads
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(threadName)s - %(message)s'
)

def worker():
    logging.debug("Starting")
    time.sleep(1)
    logging.debug("Finished")

threads = [threading.Thread(target=worker, name=f"Worker-{i}") for i in range(3)]
for t in threads:
    t.start()
for t in threads:
    t.join()

# Thread enumeration
def show_threads():
    for thread in threading.enumerate():
        print(f"Thread: {thread.name}, Alive: {thread.is_alive()}")

# Lock debugging
class DebugLock:
    def __init__(self):
        self.lock = threading.Lock()
        self.owner = None
    
    def acquire(self, blocking=True):
        thread = threading.current_thread().name
        logging.debug(f"{thread} trying to acquire lock")
        result = self.lock.acquire(blocking)
        if result:
            self.owner = thread
            logging.debug(f"{thread} acquired lock")
        return result
    
    def release(self):
        thread = threading.current_thread().name
        logging.debug(f"{thread} releasing lock")
        self.lock.release()
        self.owner = None

# Trace concurrent execution
import sys

def trace_calls(frame, event, arg):
    if event == 'call':
        code = frame.f_code
        print(f"{threading.current_thread().name}: {code.co_filename}:{code.co_name}")
    return trace_calls

# sys.settrace(trace_calls)

# Deadlock detection
import threading
from collections import defaultdict

class DeadlockDetector:
    def __init__(self):
        self.waiting_for = defaultdict(set)
        self.held_locks = defaultdict(set)
        self.lock = threading.Lock()
    
    def acquire(self, thread_id, lock_id):
        with self.lock:
            self.waiting_for[thread_id].add(lock_id)
            if self.would_deadlock(thread_id, lock_id):
                return False
            self.held_locks[thread_id].add(lock_id)
            self.waiting_for[thread_id].remove(lock_id)
            return True
    
    def would_deadlock(self, thread_id, lock_id):
        # Check if acquiring would create cycle
        # Simplified - real implementation needs graph traversal
        for other_thread, locks in self.held_locks.items():
            if lock_id in locks:
                if other_thread in self.waiting_for[thread_id]:
                    return True
        return False

# Performance profiling
import cProfile
import pstats

def profile_concurrent():
    profiler = cProfile.Profile()
    profiler.enable()
    
    # Concurrent code here
    
    profiler.disable()
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats()

# Tips:
# 1. Use logging with thread names
# 2. Add assertions for invariants
# 3. Use thread-safe data structures
# 4. Test with different thread counts
# 5. Use race condition detectors (ThreadSanitizer)
# 6. Enable Python's development mode: python -X dev

---

## Section 6: Performance Optimization & Profiling (Q81-95)

### Q81: How do you profile Python code to find bottlenecks?

**Answer:**
Use profilers to identify slow functions and optimize based on data, not assumptions.

```python
# cProfile - built-in profiler
import cProfile
import pstats

def slow_function():
    total = 0
    for i in range(1000000):
        total += i
    return total

# Profile with cProfile
profiler = cProfile.Profile()
profiler.enable()

slow_function()

profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions

# Command line profiling
# python -m cProfile -s cumulative script.py

# Line profiler - line-by-line timing
# pip install line_profiler
# @profile  # Uncomment when using kernprof
def detailed_function():
    numbers = []
    for i in range(1000):
        numbers.append(i ** 2)
    return sum(numbers)

# Run with: kernprof -l -v script.py

# Timer decorator
import time
from functools import wraps

def timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        print(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

@timer
def my_function():
    time.sleep(1)
    return "Done"

my_function()

# Timing specific code blocks
import time

start = time.perf_counter()
# Code to measure
result = sum(range(1000000))
end = time.perf_counter()
print(f"Elapsed: {end - start:.4f}s")

# timeit for accurate micro-benchmarks
import timeit

# Time a single statement
time_taken = timeit.timeit('"-".join(str(n) for n in range(100))', number=10000)
print(f"Time: {time_taken:.4f}s")

# Compare implementations
def method1():
    return [x**2 for x in range(1000)]

def method2():
    return list(map(lambda x: x**2, range(1000)))

time1 = timeit.timeit(method1, number=1000)
time2 = timeit.timeit(method2, number=1000)

print(f"Method 1: {time1:.4f}s")
print(f"Method 2: {time2:.4f}s")

# Memory profiling
# pip install memory-profiler
# @profile
def memory_intensive():
    big_list = [i for i in range(1000000)]
    return len(big_list)

# Run with: python -m memory_profiler script.py

---

### Q82: What are Python's built-in performance optimization techniques?

**Answer:**
Use list comprehensions, generators, built-in functions, and appropriate data structures.

```python
# List comprehension vs loop - comprehension is faster
# Slow
result = []
for i in range(1000):
    result.append(i ** 2)

# Fast
result = [i ** 2 for i in range(1000)]

# Generator for memory efficiency
# Memory heavy
big_list = [x ** 2 for x in range(1000000)]

# Memory efficient
big_gen = (x ** 2 for x in range(1000000))

# Use built-in functions - they're optimized in C
# Slow
total = 0
for i in range(1000):
    total += i

# Fast
total = sum(range(1000))

# Local variable lookup is faster
import math

# Slow - global lookup each iteration
def slow():
    for i in range(1000):
        x = math.sqrt(i)

# Fast - local variable
def fast():
    sqrt = math.sqrt
    for i in range(1000):
        x = sqrt(i)

# Use set for membership testing
# Slow
if item in large_list:  # O(n)
    pass

# Fast
if item in large_set:  # O(1)
    pass

# String concatenation
# Slow
result = ""
for s in strings:
    result += s  # Creates new string each time!

# Fast
result = "".join(strings)

# Dict.get() vs try/except
# Fast for common case
value = d.get(key, default)

# Fast when exceptions are rare
try:
    value = d[key]
except KeyError:
    value = default

# Use __slots__ for classes with many instances
class Point:
    __slots__ = ['x', 'y']
    
    def __init__(self, x, y):
        self.x = x
        self.y = y

# Faster than dict-based attributes

---

### Q83: How do you optimize loops and iterations?

**Answer:**
Move invariants out of loops, use appropriate iteration methods, and consider vectorization.

```python
# Move calculations out of loop
# Slow
for i in range(1000):
    result = expensive_function() * i  # Called 1000 times!

# Fast
temp = expensive_function()  # Called once
for i in range(1000):
    result = temp * i

# Use enumerate instead of range(len())
# Slow
for i in range(len(items)):
    print(i, items[i])

# Fast
for i, item in enumerate(items):
    print(i, item)

# Use zip for parallel iteration
# Slow
for i in range(len(list1)):
    process(list1[i], list2[i])

# Fast
for item1, item2 in zip(list1, list2):
    process(item1, item2)

# List comprehension vs map + lambda
# Generally faster
result = [x * 2 for x in numbers]

# Sometimes slower due to lambda overhead
result = list(map(lambda x: x * 2, numbers))

# But fast with built-in function
result = list(map(str, numbers))

# Use itertools for efficient iteration
from itertools import chain, islice

# Flatten lists
nested = [[1, 2], [3, 4], [5, 6]]
flat = list(chain.from_iterable(nested))  # Fast

# vs slow manual approach
flat = []
for sublist in nested:
    flat.extend(sublist)

# Avoid repeated attribute lookup in loops
# Slow
for i in range(1000):
    my_object.method()

# Fast
method = my_object.method
for i in range(1000):
    method()

---

### Q84: How do you optimize memory usage?

**Answer:**
Use generators, slots, appropriate data structures, and manage object lifecycle.

```python
# Generators vs lists
# Memory heavy
def get_numbers_list(n):
    return [i for i in range(n)]

numbers = get_numbers_list(1000000)  # Lots of memory

# Memory efficient
def get_numbers_gen(n):
    for i in range(n):
        yield i

numbers = get_numbers_gen(1000000)  # Tiny memory

# __slots__ for attribute storage
# Heavy
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Efficient
class PersonSlots:
    __slots__ = ['name', 'age']
    
    def __init__(self, name, age):
        self.name = name
        self.age = age

# Use array for homogeneous data
from array import array

# Heavy for numeric data
numbers_list = [1, 2, 3, 4, 5] * 1000

# Efficient
numbers_array = array('i', [1, 2, 3, 4, 5] * 1000)

# Lazy evaluation
class LazyProperty:
    def __init__(self, func):
        self.func = func
        self.name = func.__name__
    
    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = self.func(obj)
        setattr(obj, self.name, value)
        return value

class DataProcessor:
    def __init__(self, data):
        self.data = data
    
    @LazyProperty
    def expensive_result(self):
        return sum(x ** 2 for x in self.data)

# Delete large objects when done
large_data = [i for i in range(1000000)]
process(large_data)
del large_data  # Free memory immediately

# Use context managers for resource cleanup
with open('file.txt') as f:
    data = f.read()
# File automatically closed

# Avoid circular references
import weakref

class Node:
    def __init__(self, value):
        self.value = value
        self.parent = None  # Regular reference
    
    def set_parent(self, parent):
        self.parent = weakref.ref(parent)  # Weak reference

---

### Q85: How do you use NumPy for performance optimization?

**Answer:**
NumPy provides vectorized operations that are much faster than Python loops.

```python
import numpy as np
import time

# Python list - slow
python_list = list(range(1000000))
start = time.time()
result = [x ** 2 for x in python_list]
print(f"Python: {time.time() - start:.4f}s")

# NumPy array - fast
numpy_array = np.arange(1000000)
start = time.time()
result = numpy_array ** 2
print(f"NumPy: {time.time() - start:.4f}s")
# NumPy is 10-100x faster!

# Vectorized operations
# Slow - Python loop
def python_distance(x1, y1, x2, y2):
    return [(x2[i] - x1[i])**2 + (y2[i] - y1[i])**2 for i in range(len(x1))]

# Fast - NumPy vectorization
def numpy_distance(x1, y1, x2, y2):
    return (x2 - x1)**2 + (y2 - y1)**2

x1 = np.random.rand(1000000)
y1 = np.random.rand(1000000)
x2 = np.random.rand(1000000)
y2 = np.random.rand(1000000)

start = time.time()
result = numpy_distance(x1, y1, x2, y2)
print(f"NumPy: {time.time() - start:.4f}s")

# Matrix operations
matrix1 = np.random.rand(1000, 1000)
matrix2 = np.random.rand(1000, 1000)

# Fast matrix multiplication
result = np.dot(matrix1, matrix2)

# Broadcasting
array = np.array([1, 2, 3, 4, 5])
result = array + 10  # Adds 10 to each element

# Boolean indexing - fast filtering
array = np.arange(1000)
result = array[array > 500]  # Much faster than list comprehension

# In-place operations save memory
array = np.arange(1000000)
array += 1  # In-place, no new array created

---

## Section 7: Testing, Debugging & Code Quality (Q86-95)

To be continued...

(Due to response length, I'm at Q85. Would you like me to continue with the remaining 65 questions covering Testing, Modern Python Features, and Real-World Applications?)
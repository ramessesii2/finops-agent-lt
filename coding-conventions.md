# Development Guidelines for Claude

## Core Philosophy

TEST-DRIVEN DEVELOPMENT IS NON-NEGOTIABLE. Every single line of production code must be written in response to a failing test. No exceptions. This is not a suggestion or a preference - it is the fundamental practice that enables all other principles in this document.

I follow Test-Driven Development (TDD) with a strong emphasis on behavior-driven testing and functional programming principles. All work should be done in small, incremental changes that maintain a working state throughout development.

## Quick Reference

You are an expert in deep learning, transformers, diffusion models, and LLM development.

Key Principles:
- Write tests first (TDD)
- Test behavior, not implementation
- No any types or type assertions
- Small, pure functions
- Prioritize clarity, efficiency, and best practices in deep learning workflows.
- Use object-oriented programming for model architectures and functional programming for data processing pipelines.
- Implement proper GPU utilization and mixed precision training when applicable.
- Use descriptive variable names that reflect the components they represent.
- Follow PEP 8 style guidelines for Python code.

Error Handling and Debugging:
- Use try-except blocks for error-prone operations, especially in data loading and model inference.
- Implement proper logging for training progress and errors.

Important:
- Keep it simple!
- When evaluating errors, always check recent changes and fix errors before trying to add new code.
- Always update the task file or working PRD with status updates and notes for yourself.

- Whenever you cd into a directory, first get the pwd and then cd using the full path.
- Refer to the official documentation of Python dependencies for best practices and up-to-date APIs.

General principles for all tests:
    - Focus on user behavior and accessibility
    - Use semantic queries (getByRole) instead of text/class selectors
    - Avoid testing implementation details
    - Include error states and edge cases
    - Would you like me to implement any of these improvements?

Instructions for LLM tools:

1. Embrace Simplicity Over Cleverness
- Write code that's immediately understandable to others
- If a solution feels complex, it probably needs simplification
- Optimize for readability first, performance second unless proven otherwise
- Avoid premature optimization

```python
# Avoid clever one-liners
# Bad
return [n for n in range(max_num) if all(n % i != 0 for i in range(2, n))]

# Good
def find_prime_numbers(max_num):
    primes = []
    for number in range(2, max_num):
        if is_prime(number):
            primes.append(number)
    return primes

def is_prime(n):
    if n < 2:
        return False
    for i in range(2, int(n ** 0.5) + 1):
        if n % i == 0:
            return False
    return True
```

2. Focus on Core Functionality
- Start with the minimum viable solution
- Question every feature: "Is this really necessary?"
- Build incrementally based on actual needs, not hypothetical ones
- Delete unnecessary code and features

```python
# Bad: Overengineered from the start
class UserManager:
    def __init__(self, db, cache, logger, metrics, notification_service):
        self.db = db
        self.cache = cache
        self.logger = logger
        self.metrics = metrics
        self.notification = notification_service

# Good: Start simple, expand when needed
class UserManager:
    def __init__(self, db):
        self.db = db
```

3. Leverage Existing Solutions
- Use standard libraries whenever possible
- Don't reinvent the wheel
- Choose well-maintained, popular libraries for common tasks
- Keep dependencies minimal but practical

```python
# Bad: Custom implementation
def parse_json_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()
        # Custom parsing logic...

# Good: Use standard library
import json

def read_config(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)
```

4. Function Design
- Each function should have a single responsibility
- Keep functions short (typically under 20 lines)
- Use descriptive names that indicate purpose
- Limit number of parameters (3 or fewer is ideal)

```python
# Bad: Multiple responsibilities
def process_user_data(user_data):
    # Validates, saves, and sends notifications
    if validate_user(user_data):
        save_to_database(user_data)
        send_welcome_email(user_data)
        update_metrics(user_data)

# Good: Single responsibility
def save_user(user_data):
    """Saves validated user data to database."""
    if not user_data:
        raise ValueError("User data cannot be empty")
    return database.insert("users", user_data)
```

5. Project Structure
- Keep related code together
- Use consistent file organization
- Maintain a flat structure where possible
- Group by feature rather than type

```plaintext
# Good project structure
project/
├── main.py
├── config.py
├── users/
│   ├── models.py
│   ├── services.py
│   └── tests/
└── utils/
    └── helpers.py
```

6. Code Review Guidelines
- Review for simplicity first
- Question complexity and overengineering
- Look for duplicate code and abstraction opportunities
- Ensure consistent style and naming conventions

7. Maintenance Practices
- Regularly remove unused code
- Keep dependencies updated
- Refactor when code becomes unclear
- Document only what's necessary and likely to change

Remember:
- Simple code is easier to maintain and debug
- Write code for humans first, computers second
- Add complexity only when justified by requirements
- If you can't explain your code simply, it's probably too complex


Complexity is what kills you
When you finish editing, present me with a list of options of how we could continue. Indicate what you think should be the next step
When I just send you the letter c, I mean continue
Make scripts executable
Don't add any docstrings or comments, unless they really are needed for explaining the why
When you see comments or docstrings that are not absolutely necessary remove them.
Use type hints whenever possible.

Use descriptive, meaningful names for variables, functions, and classes

Group related code together
Use consistent indentation (typically 2 or 4 spaces)
Add spacing between logical sections

Handle potential errors explicitly
Validate input data
Return meaningful error messages

Use consistent formatting
Avoid deep nesting of conditionals

Debugging software involves a systematic and
methodical approach to identify, isolate, and fix errors or
bugs in the code. Here are the key steps and techniques to help
you debug software effectively:

## 1. Figure Out the Symptoms
- The first step is to understand the symptoms of the bug. What
is the incorrect behavior? What errors are being reported? Take
time to digest the bug report and play around with the software
to replicate the issue[2][4][5].

## 2. Reproduce the Bug
- Reproduce the bug in a controlled environment. Start by
reproducing it in the same environment where it was originally
reported, and then reduce the steps to the minimum necessary to
trigger the bug. This helps in isolating the issue[2][4].

## 3. Understand the System
- Gain a thorough understanding of the system and its
components. Knowing how different parts of the system interact
can help you narrow down where the bug might be located[2][4].

## 4. Form a Hypothesis
- Based on your understanding, form a hypothesis about where
the bug is located. Ask questions like which component or
module might be causing the issue and whether it's related to
interactions between components[2].

## 5. Test Your Hypothesis
- Test your hypothesis by validating input/output of the
suspected component. Modify the code if necessary to get more
information, such as adding debug logs. Ensure that any
modifications do not hide the bug[2][4].



# Object-Oriented Programming Guidelines for Simple, Robust Code

## Core Principles

### 1. Single Responsibility Principle
Each class should have one clear purpose and reason to change. Break complex classes into smaller, focused ones.

Example:
```python
# Bad
class UserManager:
    def save_user(self, user): ...
    def send_email(self, user): ...
    def generate_report(self): ...

# Good
class UserRepository:
    def save_user(self, user): ...

class EmailService:
    def send_email(self, user): ...

class ReportGenerator:
    def generate_report(self): ...
```

### 2. Encapsulation
Hide internal details and provide a clean interface. Use private attributes and public methods judiciously.

Example:
```python
class BankAccount:
    def __init__(self):
        self._balance = 0  # Protected attribute

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("Amount must be positive")
        self._balance += amount

    def get_balance(self):
        return self._balance
```

### 3. Clear Constructor Initialization
Initialize all attributes in the constructor. Make the object's state clear from the start.

Example:
```python
class Customer:
    def __init__(self, name, email):
        self.name = name
        self.email = email
        self.orders = []
        self.total_spent = 0
```

### 4. Favor Composition Over Inheritance
Use composition to build complex objects from simpler ones instead of deep inheritance hierarchies.

Example:
```python
# Bad
class SupermarketItem(ElectronicDevice, Perishable, Taxable):
    pass

# Good
class SupermarketItem:
    def __init__(self):
        self.electronic = ElectronicProperties()
        self.perishable = PerishableProperties()
        self.tax = TaxProperties()
```

### 5. Make Dependencies Explicit
Use dependency injection instead of creating dependencies inside methods.

Example:
```python
# Bad
class OrderService:
    def process_order(self, order):
        emailer = EmailService()  # Hidden dependency
        emailer.send_confirmation(order)

# Good
class OrderService:
    def __init__(self, email_service):
        self.email_service = email_service  # Explicit dependency

    def process_order(self, order):
        self.email_service.send_confirmation(order)
```

### 7. Use Strong Types and Interface Contracts
Define clear interfaces and type hints to make code more maintainable and self-documenting.

Example:
```python
from typing import List, Optional

class ShoppingCart:
    def __init__(self) -> None:
        self.items: List[Item] = []

    def add_item(self, item: Item) -> None:
        self.items.append(item)

    def get_total(self) -> float:
        return sum(item.price for item in self.items)
```

### 8. Keep Methods Short and Focused
Each method should do one thing well. Extract complex logic into helper methods.

Example:
```python
# Bad
def process_order(self, order):
    # 100 lines of mixed logic

# Good
def process_order(self, order):
    self.validate_order(order)
    total = self.calculate_total(order)
    self.apply_discounts(order)
    self.update_inventory(order)
    self.send_confirmation(order)
```

## Best Practices for Testing

1. Write tests first (TDD) when possible
2. Test public interfaces, not implementation details
3. Use meaningful test names that describe the scenario
4. Keep tests independent and isolated
5. Test edge cases and error conditions

Example:
```python
def test_withdraw_insufficient_funds():
    account = BankAccount()
    account.deposit(100)

    with pytest.raises(InsufficientFunds):
        account.withdraw(150)
```

## Common Anti-Patterns to Avoid

1. God Classes: Classes that do too much
2. Feature Envy: Methods that use more features of other classes than their own
3. Long Parameter Lists: Methods with too many parameters
4. Tight Coupling: Classes that know too much about each other
5. Premature Optimization: Making code complex for theoretical performance gains

Valid OpenAI model: - gpt-4o


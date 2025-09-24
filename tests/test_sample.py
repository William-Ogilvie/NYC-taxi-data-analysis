# pytest looks for files of the form test_*.py or *_test.py
# docs here: https://docs.pytest.org/en/stable/getting-started.html
import pytest

def func(x):
    return x + 1

def test_answer():
    assert func(3) == 5

# to execute this test in quiet reporting mode use pytest -q test_sample.py
def f():
    raise SystemExit(1)

def test_mytest():
    with pytest.raises(SystemExit):
        f()

# Pytest will look for test_ prefixed functions, ensure prefix class with Test to ensure it picks up on it
# worht noting that each test function has a unique instance of the class, so test three passes but test four fails

class TestClass:
    value = 0 
    def test_one(self):
        x = "this"
        assert "h" in x
    
    def test_two(self):
        x = "hello"
        assert hasattr(x, "check")

    def test_three(self):
        self.value += 1
        assert self.value == 1
    
    def test_four(self):
        assert self.value == 1
    
# fixtures are basically helper functions for the tests kinda

class Fruit:
    def __init__(self, name):
        self.name = name
        self.cubed = False
    
    def cube(self):
        self.cubed = True

class FruitSalad:
    def __init__(self, *fruit_bowl): # * means accept any number of arguements and store as tuple in fruit_bowl
        self.fruit = fruit_bowl
        self._cube_fruit()
    
    def _cube_fruit(self):
        for fruit in self.fruit:
            fruit.cube()

# Arrange
@pytest.fixture
def fruit_bowl():
    return [Fruit("apple"), Fruit("banana")]

def test_fruit_salad(fruit_bowl):
    # Act
    fruit_salad = FruitSalad(*fruit_bowl) # * unpacks the list into individual arguments
    # essentially pytest will call fruit_bowl() and pass the result to the test function,

    # Assert
    assert all(fruit.cubed for fruit in fruit_salad.fruit)

# This is what it would look like without the fixture
def fruit_bowl():
    return [Fruit("apple"), Fruit("banana")]


def test_fruit_salad(fruit_bowl):
    # Act
    fruit_salad = FruitSalad(*fruit_bowl)

    # Assert
    assert all(fruit.cubed for fruit in fruit_salad.fruit)


# Arrange
bowl = fruit_bowl()
test_fruit_salad(fruit_bowl=bowl)

# You can have fixtures depend on other fixtures too
# Arrange
@pytest.fixture
def first_entry():
    return "a"

# Arrange
@pytest.fixture
def order(first_entry):
    return [first_entry]

def test_string(order):
    # Act
    order.append("b")

    # Assert 
    assert order == ["a", "b"]

# the reason this is useful is that you can have multiple tests that use the same fixture
def test_int(order):
    # Act
    order.append(1)

    # Assert 
    assert order == ["a", 1]

# You can request more than one fixture at a time:

# Arrange
@pytest.fixture
def first_entry():
    return "a"


# Arrange
@pytest.fixture
def second_entry():
    return 2


# Arrange
@pytest.fixture
def order(first_entry, second_entry):
    return [first_entry, second_entry]


# Arrange
@pytest.fixture
def expected_list():
    return ["a", 2, 3.0]


def test_string(order, expected_list):
    # Act
    order.append(3.0)

    # Assert
    assert order == expected_list
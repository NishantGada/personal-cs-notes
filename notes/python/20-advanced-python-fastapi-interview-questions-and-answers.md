# Top 20 Advanced Python & FastAPI Interview Questions

## Advanced Python Questions

### 1. **Explain the Global Interpreter Lock (GIL) and its implications**

**Answer:** The GIL is a mutex (Mutual Exclusion Lock) that protects access to Python objects, preventing multiple threads from executing Python bytecode simultaneously. This means:
- CPU-bound operations don't benefit from multi-threading
- I/O-bound operations can still benefit from threading (GIL released during I/O)
- Use `multiprocessing` for CPU-bound parallelism or async/await for I/O-bound concurrency
- Libraries like NumPy release the GIL for performance-critical operations

### 2. **What are metaclasses and when would you use them?**

**Answer:** Metaclasses are classes of classes that define how classes behave. They control class creation and can modify class attributes/methods at definition time.

```python
class SingletonMeta(type):
    _instances = {}
    
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]

class Database(metaclass=SingletonMeta):
    pass
```

Use cases: ORMs (like SQLAlchemy), singletons, validation frameworks, API frameworks

### 3. **Explain descriptors and their protocol**

**Answer:** Descriptors are objects that define `__get__`, `__set__`, or `__delete__` methods, controlling attribute access.

```python
class ValidatedAttribute:
    def __init__(self, validator):
        self.validator = validator
        self.data = {}
    
    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self.data.get(id(obj))
    
    def __set__(self, obj, value):
        if not self.validator(value):
            raise ValueError("Invalid value")
        self.data[id(obj)] = value
```

Used internally by `@property`, `@staticmethod`, `@classmethod`

### 4. **What's the difference between `__new__` and `__init__`?**

**Answer:**
- `__new__`: Creates and returns the instance (called first, class method)
- `__init__`: Initializes the instance after creation

```python
class ImmutablePoint:
    def __new__(cls, x, y):
        instance = super().__new__(cls)
        instance._x = x
        instance._y = y
        return instance
    
    def __init__(self, x, y):
        # Already set in __new__
        pass
```

Use `__new__` for: immutable types, singletons, factory patterns

### 5. **Explain Python's memory management and garbage collection**

**Answer:**
- **Reference counting**: Primary mechanism, deallocates when refcount = 0
- **Cyclic GC**: Detects and collects circular references using generational collection (3 generations)
- **Memory pools**: PyMalloc manages small objects efficiently
- Use `gc` module to control collection, `weakref` for weak references

```python
import gc
gc.collect()  # Force collection
gc.get_stats()  # Get GC statistics
```

### 6. **What are coroutines and how do they differ from generators?**

**Answer:**
- **Generators**: Use `yield` to produce values (data pipeline)
- **Coroutines**: Use `async/await` for concurrent execution (cooperative multitasking)

```python
# Generator
def data_generator():
    for i in range(10):
        yield i

# Coroutine
async def fetch_data(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.json()
```

### 7. **Explain context managers and their use cases**

**Answer:** Context managers implement `__enter__` and `__exit__` for resource management.

```python
from contextlib import contextmanager

@contextmanager
def managed_resource():
    resource = acquire_resource()
    try:
        yield resource
    finally:
        release_resource(resource)

# Async version
from contextlib import asynccontextmanager

@asynccontextmanager
async def async_managed_resource():
    resource = await acquire_async_resource()
    try:
        yield resource
    finally:
        await release_async_resource(resource)
```

### 8. **What are slots and when should you use them?**

**Answer:** `__slots__` restricts instance attributes, reducing memory overhead and speeding up attribute access.

```python
class OptimizedClass:
    __slots__ = ['name', 'age', 'email']
    
    def __init__(self, name, age, email):
        self.name = name
        self.age = age
        self.email = email
```

Benefits: ~40% memory reduction, faster attribute access. Drawbacks: No dynamic attributes, no `__dict__`

### 9. **Explain the difference between deep and shallow copy**

**Answer:**
```python
import copy

original = [[1, 2], [3, 4]]

# Shallow copy: copies outer structure, inner objects shared
shallow = copy.copy(original)
shallow[0][0] = 999  # Affects original!

# Deep copy: recursively copies all objects
deep = copy.deepcopy(original)
deep[0][0] = 999  # Does NOT affect original
```

### 10. **What are Python decorators and how do they work internally?**

**Answer:** Decorators are callables that modify functions/classes using closure properties.

```python
from functools import wraps
import time

def timing_decorator(func):
    @wraps(func)  # Preserves metadata
    async def async_wrapper(*args, **kwargs):
        start = time.time()
        result = await func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    
    def sync_wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {time.time() - start:.2f}s")
        return result
    
    return async_wrapper if asyncio.iscoroutinefunction(func) else sync_wrapper
```

## Advanced FastAPI Questions

### 11. **Explain FastAPI's dependency injection system**

**Answer:** FastAPI uses function parameters to declare dependencies, resolved automatically via type annotations.

```python
from fastapi import Depends
from sqlalchemy.orm import Session

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Sub-dependency
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)):
    # Validate token and get user
    return user

@app.get("/items/")
async def read_items(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    return db.query(Item).filter(Item.owner_id == current_user.id).all()
```

### 12. **How does FastAPI handle async and sync path operations?**

**Answer:** FastAPI runs sync functions in a threadpool, async functions in the event loop.

```python
@app.get("/sync")
def sync_endpoint():
    # Runs in threadpool (blocking operations OK)
    time.sleep(1)
    return {"status": "done"}

@app.get("/async")
async def async_endpoint():
    # Runs in event loop (use await for I/O)
    await asyncio.sleep(1)
    return {"status": "done"}

# Best practice for DB operations
@app.get("/users/{user_id}")
async def get_user(user_id: int, db: Session = Depends(get_db)):
    # Use run_in_executor for sync DB calls
    loop = asyncio.get_event_loop()
    user = await loop.run_in_executor(None, db.query(User).get, user_id)
    return user
```

### 13. **Explain FastAPI's background tasks**

**Answer:** Background tasks execute after returning a response, useful for non-blocking operations.

```python
from fastapi import BackgroundTasks

async def send_email(email: str, message: str):
    # Simulated email sending
    await asyncio.sleep(2)
    print(f"Email sent to {email}")

@app.post("/register/")
async def register_user(
    user: UserCreate,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    db_user = create_user(db, user)
    background_tasks.add_task(send_email, user.email, "Welcome!")
    return db_user
```

### 14. **How do you implement custom middleware in FastAPI?**

**Answer:**
```python
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
import time

class TimingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        start_time = time.time()
        
        # Add custom headers to request
        request.state.request_id = str(uuid.uuid4())
        
        response = await call_next(request)
        
        # Add custom headers to response
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        response.headers["X-Request-ID"] = request.state.request_id
        
        return response

app.add_middleware(TimingMiddleware)
```

### 15. **Explain Pydantic validators and their types**

**Answer:**
```python
from pydantic import BaseModel, validator, root_validator, Field
from typing import Optional

class UserCreate(BaseModel):
    email: str
    password: str
    confirm_password: str
    age: Optional[int] = Field(None, ge=18, le=120)
    
    @validator('email')
    def validate_email(cls, v):
        if '@' not in v:
            raise ValueError('Invalid email')
        return v.lower()
    
    @validator('password')
    def validate_password(cls, v):
        if len(v) < 8:
            raise ValueError('Password must be 8+ characters')
        return v
    
    @root_validator
    def validate_passwords_match(cls, values):
        pw = values.get('password')
        confirm = values.get('confirm_password')
        if pw != confirm:
            raise ValueError('Passwords do not match')
        return values
```

### 16. **How do you handle database transactions in FastAPI?**

**Answer:**
```python
from contextlib import contextmanager
from sqlalchemy.exc import SQLAlchemyError

@contextmanager
def transaction_scope(db: Session):
    try:
        yield db
        db.commit()
    except SQLAlchemyError as e:
        db.rollback()
        raise
    finally:
        db.close()

@app.post("/transfer/")
async def transfer_funds(
    transfer: TransferRequest,
    db: Session = Depends(get_db)
):
    with transaction_scope(db) as session:
        from_account = session.query(Account).filter_by(id=transfer.from_id).with_for_update().first()
        to_account = session.query(Account).filter_by(id=transfer.to_id).with_for_update().first()
        
        if from_account.balance < transfer.amount:
            raise HTTPException(400, "Insufficient funds")
        
        from_account.balance -= transfer.amount
        to_account.balance += transfer.amount
        
        session.add_all([from_account, to_account])
    
    return {"status": "success"}
```

### 17. **Explain FastAPI's request validation and error handling**

**Answer:**
```python
from fastapi import HTTPException, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={
            "detail": exc.errors(),
            "body": exc.body,
            "request_id": request.state.request_id
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "message": exc.detail,
            "request_id": request.state.request_id
        }
    )
```

### 18. **How do you implement rate limiting in FastAPI?**

**Answer:**
```python
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

@app.get("/limited")
@limiter.limit("5/minute")
async def limited_endpoint(request: Request):
    return {"message": "This endpoint is rate limited"}

# Advanced: Redis-backed rate limiting
from aioredis import Redis

class RedisRateLimiter:
    def __init__(self, redis: Redis, max_requests: int, window: int):
        self.redis = redis
        self.max_requests = max_requests
        self.window = window
    
    async def is_allowed(self, key: str) -> bool:
        current = await self.redis.incr(key)
        if current == 1:
            await self.redis.expire(key, self.window)
        return current <= self.max_requests
```

### 19. **Explain FastAPI's WebSocket support**

**Answer:**
```python
from fastapi import WebSocket, WebSocketDisconnect

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
    
    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)
    
    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: int):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            await manager.broadcast(f"Client {client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(websocket)
        await manager.broadcast(f"Client {client_id} left")
```

### 20. **How do you implement API versioning in FastAPI?**

**Answer:**
```python
from fastapi import APIRouter

# Method 1: Router-based versioning
v1_router = APIRouter(prefix="/api/v1")
v2_router = APIRouter(prefix="/api/v2")

@v1_router.get("/users/")
async def get_users_v1():
    return {"version": "v1", "users": []}

@v2_router.get("/users/")
async def get_users_v2(include_deleted: bool = False):
    return {"version": "v2", "users": [], "include_deleted": include_deleted}

app.include_router(v1_router)
app.include_router(v2_router)

# Method 2: Header-based versioning
@app.get("/users/")
async def get_users(request: Request):
    version = request.headers.get("API-Version", "v1")
    if version == "v1":
        return get_users_v1_logic()
    elif version == "v2":
        return get_users_v2_logic()
    raise HTTPException(400, "Unsupported API version")
```

---

These questions cover the most critical advanced concepts you'll encounter in Python and FastAPI interviews. Focus on understanding the underlying mechanisms and when to apply each pattern!
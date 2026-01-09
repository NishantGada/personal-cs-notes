# Spring Boot Interview Master Guide (100 Q&A)

**Difficulty split:** 20 Easy · 40 Medium · 40 Hard
**Goal:** Conceptual clarity, real‑world understanding, interview‑ready explanations.
**Note:** Code snippets are intentionally minimal.

---

## 🟢 EASY (1–20)

### 1. What is Spring Boot and why was it introduced?

Spring Boot is an opinionated framework built on top of the Spring ecosystem that simplifies application development by eliminating boilerplate configuration. It was introduced to reduce the complexity of traditional Spring applications that required extensive XML or Java configuration. Boot emphasizes convention over configuration, allowing developers to focus on business logic rather than setup.

### 2. How is Spring Boot different from the Spring Framework?

Spring Framework is a comprehensive ecosystem offering modules for dependency injection, MVC, security, data access, etc. Spring Boot is not a replacement but an enhancer—it auto‑configures these modules, provides embedded servers, and offers production‑ready defaults. Spring requires configuration; Spring Boot assumes sensible defaults.

### 3. What are Spring Boot starters?

Starters are curated dependency descriptors that bundle commonly used libraries together. For example, `spring-boot-starter-web` includes Spring MVC, Jackson, and an embedded Tomcat. Starters prevent dependency mismatch and simplify dependency management.

### 4. What is auto‑configuration in Spring Boot?

Auto‑configuration automatically configures Spring beans based on classpath dependencies, existing beans, and property settings. For instance, if Spring Boot detects JPA and a DataSource, it auto‑configures `EntityManagerFactory`. This happens via conditional annotations.

### 5. What is an embedded server and why is it useful?

Spring Boot embeds servers like Tomcat, Jetty, or Undertow directly into the application. This allows applications to run as standalone JARs, simplifying deployment and eliminating the need for external application servers.

### 6. What is `@SpringBootApplication`?

It is a composite annotation combining:

* `@Configuration`
* `@EnableAutoConfiguration`
* `@ComponentScan`
  It marks the main class and triggers component scanning and auto‑configuration.

### 7. What is the default configuration file in Spring Boot?

`application.properties` or `application.yml` located in `src/main/resources`. Spring Boot automatically loads it at startup.

### 8. What is the default port of a Spring Boot application?

The default port is **8080**, configurable via `server.port`.

### 9. What is Spring Boot CLI?

Spring Boot CLI is a command‑line tool that allows rapid prototyping using Groovy scripts. It automatically resolves dependencies and runs applications without explicit build configuration.

### 10. What is Spring Initializr?

Spring Initializr is a project generation tool that creates a pre‑configured Spring Boot project with selected dependencies. It ensures best practices and proper structure.

### 11. What is dependency injection in Spring Boot?

Dependency Injection (DI) is a design pattern where dependencies are provided by the framework rather than created manually. Spring Boot uses constructor, setter, or field injection, with constructor injection being preferred.

### 12. What is `@RestController`?

It is a specialization of `@Controller` that combines `@Controller` and `@ResponseBody`, meaning all methods return serialized objects (usually JSON).

### 13. What is `@RequestMapping`?

It maps HTTP requests to controller methods based on URL patterns, HTTP methods, and headers.

### 14. What is `@ComponentScan`?

It instructs Spring where to search for annotated components like `@Component`, `@Service`, and `@Repository`.

### 15. What is Spring Boot DevTools?

DevTools improves developer productivity by enabling automatic restarts, live reload, and disabling template caching during development.

### 16. What is the difference between JAR and WAR deployment?

JAR deployment uses embedded servers and is self‑contained. WAR deployment is used when deploying to external application servers. Spring Boot favors JAR deployments.

### 17. What is `CommandLineRunner`?

It is an interface that allows execution of logic after the application context is loaded, commonly used for initialization tasks.

### 18. What is `@Value` annotation?

It injects values from properties files, environment variables, or expressions into fields or method parameters.

### 19. What is Actuator?

Spring Boot Actuator provides production‑ready endpoints for monitoring and managing applications, such as health checks and metrics.

### 20. What is Spring Boot’s default logging framework?

Spring Boot uses **Logback** by default, with SLF4J as the logging facade.

---

## 🟡 MEDIUM (21–60)

### 21. How does Spring Boot auto‑configuration work internally?

Auto‑configuration relies on `@EnableAutoConfiguration`, which imports configuration classes listed in `spring.factories`. These configurations use conditional annotations to decide whether beans should be created.

### 22. What are conditional annotations?

Annotations like `@ConditionalOnClass`, `@ConditionalOnMissingBean`, and `@ConditionalOnProperty` control bean creation based on runtime conditions.

### 23. How does Spring Boot manage externalized configuration?

Spring Boot supports property sources such as properties files, YAML, environment variables, command‑line arguments, and config servers, following a defined precedence order.

### 24. Difference between `application.properties` and `application.yml`?

Both serve the same purpose. YAML supports hierarchical configuration and is more readable for complex structures, while properties files are simpler and flat.

### 25. What is Spring Boot Profiles?

Profiles allow environment‑specific configurations (dev, test, prod). They enable conditional bean loading and property segregation.

### 26. How does Spring Boot handle exception management?

Spring Boot uses `@ControllerAdvice` and `@ExceptionHandler` for centralized exception handling, allowing consistent API responses.

### 27. What is `@ConfigurationProperties`?

It binds external configuration to strongly‑typed Java objects, improving maintainability compared to scattered `@Value` usage.

### 28. Difference between `@Component`, `@Service`, and `@Repository`?

They are semantically different stereotypes. `@Repository` adds exception translation, `@Service` denotes business logic, while `@Component` is generic.

### 29. How does Spring Boot integrate with databases?

Spring Boot auto‑configures DataSource, JPA, Hibernate, and transaction management based on dependencies and properties.

### 30. What is Spring Data JPA?

Spring Data JPA abstracts data access by generating repository implementations automatically, reducing boilerplate CRUD code.

### 31. How does Spring Boot handle transactions?

Transactions are managed using `@Transactional`, which uses AOP proxies to ensure atomicity and consistency.

### 32. Difference between `CrudRepository` and `JpaRepository`?

`JpaRepository` extends `CrudRepository` and adds pagination, sorting, and batch operations.

### 33. How does Spring Boot secure applications?

Spring Boot integrates with Spring Security, offering authentication, authorization, CSRF protection, and OAuth2 support.

### 34. What is CSRF and how does Spring Boot handle it?

CSRF is a security attack exploiting authenticated sessions. Spring Security enables CSRF protection by default for state‑changing requests.

### 35. How does Spring Boot handle JSON serialization?

Spring Boot uses Jackson by default to convert Java objects to JSON and vice versa.

### 36. What is Spring Boot Actuator health check?

Health endpoints expose application health indicators like database connectivity and disk space.

### 37. What is HikariCP?

HikariCP is the default high‑performance JDBC connection pool in Spring Boot.

### 38. How does Spring Boot support RESTful APIs?

It provides annotations like `@RestController`, `@GetMapping`, and content negotiation for RESTful design.

### 39. What is content negotiation?

It allows clients to request specific response formats (JSON/XML) via headers like `Accept`.

### 40. How does Spring Boot handle validation?

It integrates with Bean Validation (JSR‑380) using annotations like `@NotNull` and `@Size`.

### 41. What is Spring Boot caching abstraction?

It provides a uniform caching API supporting providers like EhCache, Redis, and Caffeine.

### 42. How does Spring Boot support async processing?

Using `@Async`, methods can run in separate threads managed by Spring’s task executor.

### 43. What is `RestTemplate` and why is it deprecated?

`RestTemplate` is a synchronous HTTP client. It’s deprecated in favor of `WebClient`, which supports reactive, non‑blocking calls.

### 44. What is WebClient?

WebClient is a reactive HTTP client built on Project Reactor, enabling non‑blocking communication.

### 45. Difference between MVC and WebFlux?

MVC is synchronous and thread‑per‑request. WebFlux is reactive and event‑driven, handling high concurrency efficiently.

### 46. How does Spring Boot handle logging configuration?

It supports external logging configuration via Logback or Log4j2 files and properties.

### 47. What is Spring Boot’s default error handling?

Spring Boot provides a `/error` endpoint and whitelabel error page for standardized error responses.

### 48. What is Spring Boot testing support?

Spring Boot offers annotations like `@SpringBootTest` and auto‑configured test slices.

### 49. Difference between unit tests and integration tests in Spring Boot?

Unit tests isolate components; integration tests load application context and verify component interaction.

### 50. What is test slicing?

Test slicing loads only required beans (e.g., `@WebMvcTest`) to speed up tests.

### 51. How does Spring Boot manage environment variables?

Environment variables override configuration properties following precedence rules.

### 52. What is Spring Boot banner?

It’s a customizable ASCII art displayed during application startup.

### 53. How does Spring Boot handle multipart file uploads?

It auto‑configures multipart resolvers for handling file uploads via REST endpoints.

### 54. What is `@EnableScheduling`?

It enables scheduled task execution using annotations like `@Scheduled`.

### 55. How does Spring Boot handle graceful shutdown?

It supports graceful shutdown via actuator and server settings, allowing active requests to complete.

### 56. What is Spring Boot’s default transaction manager?

It auto‑selects a transaction manager based on available DataSource or JPA setup.

### 57. How does Spring Boot support internationalization (i18n)?

It uses message bundles and locale resolvers to support multiple languages.

### 58. What is Spring Boot’s fat JAR?

A fat JAR contains application code, dependencies, and embedded server in a single file.

### 59. What is Spring Boot’s startup lifecycle?

It initializes environment, creates application context, applies auto‑configurations, and starts embedded server.

### 60. What is Spring Boot Admin?

It is a community tool for managing and monitoring Spring Boot applications via UI.

---

## 🔴 HARD (61–100)

### 61. Explain Spring Boot’s auto‑configuration ordering

Auto‑configuration classes are applied in a defined order using `@AutoConfigureBefore` and `@AutoConfigureAfter`, ensuring correct bean initialization.

### 62. How does Spring Boot use classpath scanning efficiently?

It uses metadata files to avoid loading unnecessary classes, improving startup performance.

### 63. What are `spring.factories` and `META-INF`?

They define auto‑configuration entries and extension points used during application startup.

### 64. How does Spring Boot differ from Micronaut or Quarkus?

Spring Boot relies heavily on runtime reflection, while Micronaut and Quarkus emphasize compile‑time DI for faster startup.

### 65. What is context hierarchy in Spring Boot?

It allows parent‑child application contexts, useful in complex modular applications.

### 66. Explain how Spring Boot handles circular dependencies

Spring resolves circular dependencies via proxies, but constructor injection prevents them by design.

### 67. How does Spring Boot optimize startup time?

Lazy initialization, classpath scanning optimizations, and conditional bean loading reduce startup cost.

### 68. What is `@SpringBootTest` internally doing?

It boots the full application context, mimicking production startup behavior for testing.

### 69. Explain reactive backpressure in WebFlux

Backpressure ensures consumers are not overwhelmed by producers, managed via reactive streams.

### 70. How does Spring Boot integrate with Kubernetes?

Through Actuator, health probes, config maps, secrets, and container‑friendly startup.

### 71. What is Spring Boot’s role in microservices architecture?

It provides lightweight, independently deployable services with embedded servers and cloud integrations.

### 72. How does Spring Boot handle distributed tracing?

It integrates with tools like OpenTelemetry and Sleuth to propagate trace IDs.

### 73. What is Spring Boot’s memory footprint concern?

Reflection and auto‑configurations increase memory usage compared to native frameworks.

### 74. Explain Spring Boot native images

Using GraalVM, Spring Boot apps can be compiled into native binaries for fast startup.

### 75. What is AOT processing in Spring Boot?

Ahead‑of‑Time processing generates metadata at build time to reduce runtime reflection.

### 76. How does Spring Boot manage thread pools?

It auto‑configures thread pools for web, async, and scheduling tasks.

### 77. What is `ApplicationContextInitializer`?

It allows programmatic customization of application context before it is refreshed.

### 78. How does Spring Boot handle multi‑tenancy?

Via schema‑based, database‑based, or discriminator‑based strategies using Hibernate.

### 79. Explain Spring Boot’s security filter chain

Spring Security applies filters in a specific order to handle authentication and authorization.

### 80. What is Spring Boot’s event system?

It publishes lifecycle and custom events that listeners can react to asynchronously.

### 81. How does Spring Boot handle API versioning?

Via URL paths, headers, or content negotiation strategies.

### 82. What are custom starters?

Reusable dependency bundles that encapsulate common auto‑configuration logic.

### 83. Explain `@Import` vs auto‑configuration

`@Import` explicitly loads configurations; auto‑configuration loads conditionally.

### 84. How does Spring Boot manage secrets securely?

Using environment variables, vaults, and external config servers.

### 85. What is Spring Cloud Config?

Centralized configuration management for distributed systems.

### 86. Explain circuit breakers in Spring Boot

Implemented via Resilience4j to prevent cascading failures.

### 87. How does Spring Boot handle rate limiting?

Via API gateways or libraries like Bucket4j.

### 88. Explain Spring Boot’s request lifecycle

From dispatcher servlet to controller, service, and response serialization.

### 89. What is Spring Boot’s default exception translation?

Converts low‑level exceptions into consistent, meaningful responses.

### 90. How does Spring Boot manage schema migrations?

Using tools like Flyway or Liquibase.

### 91. What is Spring Boot’s observability model?

Metrics, logs, and traces exposed via Actuator and Micrometer.

### 92. Explain Spring Boot’s classloader structure

Custom classloaders isolate dependencies in fat JARs.

### 93. How does Spring Boot handle hot reload?

Via DevTools classloader separation.

### 94. What is Spring Boot’s default JSON mapper customization?

Jackson ObjectMapper is auto‑configured but customizable via beans.

### 95. How does Spring Boot handle concurrency issues?

Through thread safety, transactions, and proper bean scopes.

### 96. What is Spring Boot’s dependency version alignment?

Managed via Spring Boot BOM to ensure compatibility.

### 97. Explain Spring Boot’s actuator security

Endpoints can be selectively exposed and secured.

### 98. What is Spring Boot’s approach to backward compatibility?

Strong versioning guarantees and deprecation policies.

### 99. How does Spring Boot support API documentation?

Via OpenAPI and Swagger integrations.

### 100. What makes Spring Boot production‑ready?

Auto‑configuration, monitoring, security, scalability, and cloud readiness.

---

## 📦 Recommended Sharing Format

**Primary:** Markdown (`.md`) → Exportable to **PDF / DOCX via Pandoc**
**Alternate:** Notion / GitHub README
**Why:** ATS‑safe, version‑controlled, printable, and easily extensible

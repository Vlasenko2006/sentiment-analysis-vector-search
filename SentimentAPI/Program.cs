using Microsoft.AspNetCore.Diagnostics.HealthChecks;
using Microsoft.Extensions.Diagnostics.HealthChecks;
using SentimentAPI.Services;
using System.Text.Json;
using System.Text.Json.Serialization;

var builder = WebApplication.CreateBuilder(args);

// Port configuration handled by ASPNETCORE_URLS environment variable
// No need for ConfigureKestrel - prevents port binding conflicts

// Configure JSON serialization options
builder.Services.AddControllers()
    .AddJsonOptions(options =>
    {
        options.JsonSerializerOptions.PropertyNamingPolicy = JsonNamingPolicy.CamelCase;
        options.JsonSerializerOptions.DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull;
        options.JsonSerializerOptions.WriteIndented = true;
    });

builder.Services.AddEndpointsApiExplorer();
builder.Services.AddSwaggerGen(c =>
{
    c.SwaggerDoc("v1", new() { 
        Title = "Sentiment Analysis API", 
        Version = "v1",
        Description = ".NET API Gateway for Python-based Sentiment Analysis Service"
    });
});

// Configure CORS
builder.Services.AddCors(options =>
{
    options.AddPolicy("AllowAll", builder =>
    {
        builder.AllowAnyOrigin()
               .AllowAnyMethod()
               .AllowAnyHeader();
    });
});

// Register HTTP Client for Python Service
builder.Services.AddHttpClient<IPythonServiceClient, PythonServiceClient>(client =>
{
    var pythonServiceUrl = builder.Configuration["PythonServiceUrl"] ?? "http://python-service:8000";
    client.BaseAddress = new Uri(pythonServiceUrl);
    client.Timeout = TimeSpan.FromMinutes(10); // Long timeout for analysis
});

// Add Health Checks
builder.Services.AddHealthChecks()
    .AddCheck<PythonServiceHealthCheck>("python_service");

// Configure logging
builder.Logging.ClearProviders();
builder.Logging.AddConsole();
builder.Logging.AddDebug();

var app = builder.Build();

// Configure the HTTP request pipeline
if (app.Environment.IsDevelopment())
{
    app.UseSwagger();
    app.UseSwaggerUI();
}

// Enable Swagger in production for API documentation
app.UseSwagger();
app.UseSwaggerUI(c =>
{
    c.SwaggerEndpoint("/swagger/v1/swagger.json", "Sentiment Analysis API v1");
    c.RoutePrefix = "swagger";
});

app.UseCors("AllowAll");

// Add request/response logging middleware
app.UseMiddleware<SentimentAPI.Middleware.RequestResponseLoggingMiddleware>();

app.UseAuthorization();

app.MapControllers();

// Root endpoint
app.MapGet("/", () => Results.Ok(new
{
    service = "Sentiment Analysis .NET API Gateway",
    version = "1.0.0",
    status = "running",
    endpoints = new
    {
        swagger = "/swagger",
        health = "/health",
        analyze = "/api/sentiment/analyze",
        status = "/api/sentiment/status/{jobId}",
        results = "/api/sentiment/results/{jobId}"
    }
}));

// Health check endpoint
app.MapHealthChecks("/health", new HealthCheckOptions
{
    ResponseWriter = async (context, report) =>
    {
        context.Response.ContentType = "application/json";
        var result = JsonSerializer.Serialize(new
        {
            status = report.Status.ToString(),
            timestamp = DateTime.UtcNow,
            checks = report.Entries.Select(e => new
            {
                name = e.Key,
                status = e.Value.Status.ToString(),
                description = e.Value.Description,
                duration = e.Value.Duration.TotalMilliseconds
            })
        }, new JsonSerializerOptions { WriteIndented = true });
        await context.Response.WriteAsync(result);
    }
});

app.Logger.LogInformation("🚀 Sentiment Analysis .NET API Gateway starting...");
app.Logger.LogInformation("📍 Listening on configured URLs (check ASPNETCORE_URLS)");
app.Logger.LogInformation("🔗 Python Service URL: {PythonServiceUrl}", 
    builder.Configuration["PythonServiceUrl"] ?? "http://python-service:8000");

app.Run();

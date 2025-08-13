# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import uuid # For unique session IDs
from dotenv import load_dotenv

from google.adk.agents import LlmAgent
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.genai import types
import os # Required for path operations

# --- OpenAPI Tool Imports ---
from google.adk.tools.openapi_tool.openapi_spec_parser.openapi_toolset import OpenAPIToolset

# --- MCP Tool Imports ---
from google.adk.tools.mcp_tool.mcp_toolset import MCPToolset, StdioConnectionParams, StdioServerParameters

# --- Load Environment Variables ---
load_dotenv()

# Obtener la API key de Google desde las variables de entorno
google_api_key = os.getenv('GOOGLE_API_KEY')

# Verificar que la API key esté disponible
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY no encontrada en las variables de entorno. Asegúrate de que esté definida en tu archivo .env")

# Configurar la API key como variable de entorno para que Google ADK la use
os.environ['GOOGLE_API_KEY'] = google_api_key

# --- Constants ---
APP_NAME = "combined_openapi_mcp_app"
USER_ID = "user_combined_1"
SESSION_ID = f"session_combined_{uuid.uuid4()}"
AGENT_NAME = "combined_assistant_agent"
GEMINI_MODEL = "gemini-2.5-flash"

# --- MCP Configuration ---
TARGET_FOLDER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "/home/cetec/AIProjects/adk_openapi/adk_test")

# --- OpenAPI Specification ---
openapi_spec_string = """
{
  "openapi": "3.0.3",
  "info": {
    "title": "ReqRes API",
    "description": "A hosted REST-API ready to respond to your AJAX requests. Free fake REST API for testing and prototyping.",
    "version": "1.0.0",
    "contact": {
      "name": "ReqRes",
      "url": "https://reqres.in"
    },
    "license": {
      "name": "MIT",
      "url": "https://opensource.org/licenses/MIT"
    }
  },
  "servers": [
    {
      "url": "https://reqres.in/api",
      "description": "ReqRes.in API server"
    }
  ],
  "paths": {
    "/users": {
      "get": {
        "summary": "List Users",
        "description": "Get a paginated list of users",
        "tags": ["Users"],
        "parameters": [
          {
            "name": "page",
            "in": "query",
            "required": false,
            "description": "Page number (default: 1)",
            "schema": {
              "type": "integer",
              "minimum": 1,
              "default": 1
            }
          },
          {
            "name": "per_page",
            "in": "query",
            "required": false,
            "description": "Number of users per page (default: 6)",
            "schema": {
              "type": "integer",
              "minimum": 1,
              "maximum": 12,
              "default": 6
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/UsersListResponse"
                }
              }
            }
          }
        }
      },
      "post": {
        "summary": "Create User",
        "description": "Create a new user",
        "tags": ["Users"],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/CreateUserRequest"
              }
            }
          }
        },
        "responses": {
          "201": {
            "description": "User created successfully",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/CreateUserResponse"
                }
              }
            }
          }
        }
      }
    },
    "/users/{id}": {
      "get": {
        "summary": "Get Single User",
        "description": "Get a single user by ID",
        "tags": ["Users"],
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "required": true,
            "description": "User ID",
            "schema": {
              "type": "integer"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SingleUserResponse"
                }
              }
            }
          },
          "404": {
            "description": "User not found",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorResponse"
                }
              }
            }
          }
        }
      },
      "put": {
        "summary": "Update User",
        "description": "Update an existing user",
        "tags": ["Users"],
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "required": true,
            "description": "User ID",
            "schema": {
              "type": "integer"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/UpdateUserRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "User updated successfully",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/UpdateUserResponse"
                }
              }
            }
          }
        }
      },
      "patch": {
        "summary": "Partial Update User",
        "description": "Partially update an existing user",
        "tags": ["Users"],
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "required": true,
            "description": "User ID",
            "schema": {
              "type": "integer"
            }
          }
        ],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/UpdateUserRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "User updated successfully",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/UpdateUserResponse"
                }
              }
            }
          }
        }
      },
      "delete": {
        "summary": "Delete User",
        "description": "Delete a user by ID",
        "tags": ["Users"],
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "required": true,
            "description": "User ID",
            "schema": {
              "type": "integer"
            }
          }
        ],
        "responses": {
          "204": {
            "description": "User deleted successfully"
          }
        }
      }
    },
    "/unknown": {
      "get": {
        "summary": "List Resources",
        "description": "Get a list of unknown resources (colors)",
        "tags": ["Resources"],
        "parameters": [
          {
            "name": "page",
            "in": "query",
            "required": false,
            "description": "Page number",
            "schema": {
              "type": "integer",
              "minimum": 1,
              "default": 1
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ResourcesListResponse"
                }
              }
            }
          }
        }
      }
    },
    "/unknown/{id}": {
      "get": {
        "summary": "Get Single Resource",
        "description": "Get a single resource by ID",
        "tags": ["Resources"],
        "parameters": [
          {
            "name": "id",
            "in": "path",
            "required": true,
            "description": "Resource ID",
            "schema": {
              "type": "integer"
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Success",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/SingleResourceResponse"
                }
              }
            }
          },
          "404": {
            "description": "Resource not found",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorResponse"
                }
              }
            }
          }
        }
      }
    },
    "/register": {
      "post": {
        "summary": "Register User",
        "description": "Register a new user account",
        "tags": ["Authentication"],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/RegisterRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Registration successful",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/RegisterResponse"
                }
              }
            }
          },
          "400": {
            "description": "Registration failed",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorResponse"
                }
              }
            }
          }
        }
      }
    },
    "/login": {
      "post": {
        "summary": "Login User",
        "description": "Authenticate user and get token",
        "tags": ["Authentication"],
        "requestBody": {
          "required": true,
          "content": {
            "application/json": {
              "schema": {
                "$ref": "#/components/schemas/LoginRequest"
              }
            }
          }
        },
        "responses": {
          "200": {
            "description": "Login successful",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/LoginResponse"
                }
              }
            }
          },
          "400": {
            "description": "Login failed",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/ErrorResponse"
                }
              }
            }
          }
        }
      }
    },
    "/users/{delay}": {
      "get": {
        "summary": "Delayed Response",
        "description": "Get users list with artificial delay (for testing purposes)",
        "tags": ["Testing"],
        "parameters": [
          {
            "name": "delay",
            "in": "path",
            "required": true,
            "description": "Delay in seconds",
            "schema": {
              "type": "integer",
              "minimum": 1
            }
          }
        ],
        "responses": {
          "200": {
            "description": "Success with delay",
            "content": {
              "application/json": {
                "schema": {
                  "$ref": "#/components/schemas/UsersListResponse"
                }
              }
            }
          }
        }
      }
    }
  },
  "components": {
    "schemas": {
      "User": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer",
            "description": "User ID"
          },
          "email": {
            "type": "string",
            "format": "email",
            "description": "User email address"
          },
          "first_name": {
            "type": "string",
            "description": "User first name"
          },
          "last_name": {
            "type": "string",
            "description": "User last name"
          },
          "avatar": {
            "type": "string",
            "format": "uri",
            "description": "User avatar URL"
          }
        }
      },
      "Resource": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer",
            "description": "Resource ID"
          },
          "name": {
            "type": "string",
            "description": "Resource name"
          },
          "year": {
            "type": "integer",
            "description": "Resource year"
          },
          "color": {
            "type": "string",
            "description": "Resource color code"
          },
          "pantone_value": {
            "type": "string",
            "description": "Pantone color value"
          }
        }
      },
      "Support": {
        "type": "object",
        "properties": {
          "url": {
            "type": "string",
            "format": "uri",
            "description": "Support URL"
          },
          "text": {
            "type": "string",
            "description": "Support text"
          }
        }
      },
      "UsersListResponse": {
        "type": "object",
        "properties": {
          "page": {
            "type": "integer",
            "description": "Current page number"
          },
          "per_page": {
            "type": "integer",
            "description": "Number of items per page"
          },
          "total": {
            "type": "integer",
            "description": "Total number of items"
          },
          "total_pages": {
            "type": "integer",
            "description": "Total number of pages"
          },
          "data": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/User"
            }
          },
          "support": {
            "$ref": "#/components/schemas/Support"
          }
        }
      },
      "SingleUserResponse": {
        "type": "object",
        "properties": {
          "data": {
            "$ref": "#/components/schemas/User"
          },
          "support": {
            "$ref": "#/components/schemas/Support"
          }
        }
      },
      "ResourcesListResponse": {
        "type": "object",
        "properties": {
          "page": {
            "type": "integer",
            "description": "Current page number"
          },
          "per_page": {
            "type": "integer",
            "description": "Number of items per page"
          },
          "total": {
            "type": "integer",
            "description": "Total number of items"
          },
          "total_pages": {
            "type": "integer",
            "description": "Total number of pages"
          },
          "data": {
            "type": "array",
            "items": {
              "$ref": "#/components/schemas/Resource"
            }
          },
          "support": {
            "$ref": "#/components/schemas/Support"
          }
        }
      },
      "SingleResourceResponse": {
        "type": "object",
        "properties": {
          "data": {
            "$ref": "#/components/schemas/Resource"
          },
          "support": {
            "$ref": "#/components/schemas/Support"
          }
        }
      },
      "CreateUserRequest": {
        "type": "object",
        "required": ["name", "job"],
        "properties": {
          "name": {
            "type": "string",
            "description": "User name"
          },
          "job": {
            "type": "string",
            "description": "User job title"
          }
        }
      },
      "CreateUserResponse": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "User name"
          },
          "job": {
            "type": "string",
            "description": "User job title"
          },
          "id": {
            "type": "string",
            "description": "Generated user ID"
          },
          "createdAt": {
            "type": "string",
            "format": "date-time",
            "description": "Creation timestamp"
          }
        }
      },
      "UpdateUserRequest": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "User name"
          },
          "job": {
            "type": "string",
            "description": "User job title"
          }
        }
      },
      "UpdateUserResponse": {
        "type": "object",
        "properties": {
          "name": {
            "type": "string",
            "description": "User name"
          },
          "job": {
            "type": "string",
            "description": "User job title"
          },
          "updatedAt": {
            "type": "string",
            "format": "date-time",
            "description": "Update timestamp"
          }
        }
      },
      "RegisterRequest": {
        "type": "object",
        "required": ["email", "password"],
        "properties": {
          "email": {
            "type": "string",
            "format": "email",
            "description": "User email address"
          },
          "password": {
            "type": "string",
            "description": "User password"
          }
        }
      },
      "RegisterResponse": {
        "type": "object",
        "properties": {
          "id": {
            "type": "integer",
            "description": "User ID"
          },
          "token": {
            "type": "string",
            "description": "Authentication token"
          }
        }
      },
      "LoginRequest": {
        "type": "object",
        "required": ["email", "password"],
        "properties": {
          "email": {
            "type": "string",
            "format": "email",
            "description": "User email address"
          },
          "password": {
            "type": "string",
            "description": "User password"
          }
        }
      },
      "LoginResponse": {
        "type": "object",
        "properties": {
          "token": {
            "type": "string",
            "description": "Authentication token"
          }
        }
      },
      "ErrorResponse": {
        "type": "object",
        "properties": {
          "error": {
            "type": "string",
            "description": "Error message"
          }
        }
      }
    }
  }
}
"""

# --- Create Toolsets ---
# OpenAPI Toolset
users_toolset = OpenAPIToolset(
    spec_str=openapi_spec_string,
    spec_str_type='json',
)

# MCP Toolset
filesystem_toolset = MCPToolset(
    connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(
            command='npx',
            args=[
                "-y",
                "@modelcontextprotocol/server-filesystem",
                os.path.abspath(TARGET_FOLDER_PATH),
            ],
        ),
    ),
)

# --- Combined Agent Definition ---
root_agent = LlmAgent(
    name=AGENT_NAME,
    model=GEMINI_MODEL,
    tools=[users_toolset, filesystem_toolset],  # ¡AQUÍ ESTÁN AMBOS TOOLSETS!
    instruction="""You are a versatile assistant that can:

    1. MANAGE WEATHER via API:
       - List available users and resources
       - Create new users
       - Update existing users
       - Get details for specific users
       - List resources and their details
       - Register and login users
       - Handle delayed responses for testing purposes
       - Manage weather alerts (if applicable)

    2. MANAGE FILES in the filesystem:
       - List directories and files
       - Read file contents
       - Navigate through folders
    When there is a request about users use the users_tools, without API KEY or any other credential.
    Remember the users actions dont require authentication.
    When the user asks about files, folders, or filesystem operations, use the filesystem tools.
    """,
    description="A combined assistant that manages both users via API and files via filesystem."
)

# --- Session and Runner Setup ---
async def setup_session_and_runner():
    session_service = InMemorySessionService()
    runner = Runner(
        agent=root_agent,
        app_name=APP_NAME,
        session_service=session_service,
    )
    await session_service.create_session(
        app_name=APP_NAME,
        user_id=USER_ID,
        session_id=SESSION_ID,
    )
    return runner

# --- Agent Interaction Function ---
async def call_combined_agent_async(query, runner):
    print(f"\n--- Combined Agent Query ---")
    print(f"Query: {query}")

    content = types.Content(role='user', parts=[types.Part(text=query)])
    final_response_text = "Agent did not provide a final text response."
    
    try:
        async for event in runner.run_async(
            user_id=USER_ID, 
            session_id=SESSION_ID, 
            new_message=content
        ):
            if event.get_function_calls():
                call = event.get_function_calls()[0]
                print(f"  🔧 Agent Action: Called '{call.name}' with args {call.args}")
            elif event.get_function_responses():
                response = event.get_function_responses()[0]
                print(f"  ✅ Tool Response: Function '{response.name}' completed")
            elif event.is_final_response() and event.content and event.content.parts:
                final_response_text = event.content.parts[0].text.strip()

        print(f"🤖 Agent Final Response: {final_response_text}")

    except Exception as e:
        print(f"❌ Error during agent run: {e}")
        import traceback
        traceback.print_exc()
    print("-" * 50)

# --- Run Examples ---
async def run_combined_example():
    runner = await setup_session_and_runner()

    # Test Pet Store API functionality
    print("=== TESTING PET STORE API ===")
    await call_combined_agent_async("Show me available pets in the store", runner)
    await call_combined_agent_async("Add a new cat named 'Whiskers' to the store", runner)
    await call_combined_agent_async("Get details for pet ID 456", runner)
    
    # Test Filesystem functionality  
    print("\n=== TESTING FILESYSTEM ===")
    await call_combined_agent_async("List the files in the current directory", runner)
    await call_combined_agent_async("What files are available to read?", runner)
    
    # Test mixed functionality
    print("\n=== TESTING MIXED FUNCTIONALITY ===")
    await call_combined_agent_async("Create a pet named 'Buddy' and then show me what files I have", runner)

# --- Execute ---
if __name__ == "__main__":
    print("🚀 Executing Combined OpenAPI + MCP Agent Example...")
    try:
        asyncio.run(run_combined_example())
    except RuntimeError as e:
        if "cannot be called from a running event loop" in str(e):
            print("ℹ️  Cannot run asyncio.run from a running event loop (e.g., Jupyter/Colab).")
        else:
            raise e
    print("✅ Combined agent example finished.")
# Copyright (c) Facebook, Inc. and its affiliates.
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
add_subdirectory(prestosql)

# Required to build the parser.
#string(APPEND CMAKE_CXX_FLAGS " -fbracket-depth=1024 -funsigned-char")
string(APPEND CMAKE_CXX_FLAGS " -Werror=misleading-indentation -funsigned-char")

# Needed to make #include "parser.h" work.
include_directories(/home/oerling/sql/parser/cpp)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/prestosql/generated)

add_library(velox_parse_expression Expressions.cpp)
target_link_libraries(velox_parse_expression velox_type velox_parse_utils
                      velox_expression)

add_library(velox_parse_parser ExpressionsParser.cpp QueryPlanner.cpp)
target_link_libraries(velox_parse_parser velox_parse_expression velox_type
                      velox_duckdb_parser velox_parse_prestosql)

add_library(velox_parse_utils TypeResolver.cpp)
target_link_libraries(velox_parse_utils velox_buffer velox_type velox_vector
                      velox_function_registry)

if(${VELOX_BUILD_TESTING})
  add_subdirectory(tests)
endif()

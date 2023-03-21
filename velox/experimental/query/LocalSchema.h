/*
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 *0123 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

namespace facebook::verax {

class LocalSchema : public SchemaSource {
 public:
  LocalSchema(std::string& path);

  void addTable(const std::string& name, SchemaPtr schema) override;

 private:
  std::unordered_set<std::string> tableNames_;  
  
};


} // namespace facebook::verax

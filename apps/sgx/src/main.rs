/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

extern crate tvm_graph_rt;

use std::{
    convert::TryFrom as _,
    io::{Read as _, Write as _},
    time::Instant,
    process,
};

fn main() {
    let syslib = tvm_graph_rt::SystemLibModule::default();

    let graph_json = include_str!(concat!(env!("OUT_DIR"), "/graph.json"));
    //let graph_json = include_str!("/home/svj5489/ML/torch_md/bert_12/graph.json");
    let params_bytes = include_bytes!(concat!(env!("OUT_DIR"), "/params.bin"));
    //let params_bytes = include_bytes!("/home/svj5489/ML/torch_md/bert_12/params.bin");
    let params = tvm_graph_rt::load_param_dict(params_bytes).unwrap();

    let graph = tvm_graph_rt::Graph::try_from(graph_json).unwrap();
    let mut exec = tvm_graph_rt::GraphExecutor::new(graph, &syslib).unwrap();
    exec.load_params(params);
    let now = Instant::now();
    let mut i = 0;

    while i < 30 {
        exec.run();
        i=i+1;
    }
    let elapsed_time = now.elapsed();
    
    println!("infer finishes, time: {:.6}", elapsed_time.as_secs() as f64
           + elapsed_time.subsec_nanos() as f64 * 1e-9);
}

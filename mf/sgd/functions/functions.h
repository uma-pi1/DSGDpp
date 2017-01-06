//    Copyright 2017 Rainer Gemulla
// 
//    Licensed under the Apache License, Version 2.0 (the "License");
//    you may not use this file except in compliance with the License.
//    You may obtain a copy of the License at
// 
//        http://www.apache.org/licenses/LICENSE-2.0
// 
//    Unless required by applicable law or agreed to in writing, software
//    distributed under the License is distributed on an "AS IS" BASIS,
//    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//    See the License for the specific language governing permissions and
//    limitations under the License.
/* @file
 *
 * Concepts for SGD functions.
 */
#ifndef SGD_FUNCTIONS_FUNCTIONS_H
#define SGD_FUNCTIONS_FUNCTIONS_H

/** An update function performs a single SGD step on the data. It takes as input a data point
 * and the corresponding row and column factors; these factors are directly updated. The update
 * function may also update factors in different rows and columns, although this is usually
 * not required.
 *
 * \par Methods:
 * <table>
 * <tr><th>Signature</th><th>Description</th></tr>
 * <tr>
 *   <td>void operator()(FactorizationData& data,
 *     const unsigned i, const unsigned j,	const double x,
 *     const double eps)
 *   </td>
 *   <td>Performs an SGD step on matrix entry (i,j). Here, x refers to the value at (i,j)
 *     and eps is the current step size. The data object contain the current set of factors,
 *    which are updated by this method.
 *   </td>
 * </tr>
 * </table>
 *
 * @see mf::Sgd
 * @see RegularizeConcept
 */
class UpdateConcept {
};

/** An regularize function performs a single GD or SGD step on the factors. It usually does
 * not access data, although we allow such accesses when needed.
 *
 * \par Methods:
 * <table>
 * <tr><th>Signature</th><th>Description</th></tr>
 * <tr><td>void operator()(FactorizationData& job, const double eps)</td>
 * <td>Performs a GD or SGD step on the factors. Here, eps refers to the current step size.</td></tr>
 * <tr><td>bool rescaleStratumStepsize()</td>
 * <td>True, if the stepsize should be reduced when running on a stratum in DSGD.</td></tr>
 * </table>
 *
 * @see mf::Sgd
 * @see UpdateConcept
 */
class RegularizeConcept {
};



#endif

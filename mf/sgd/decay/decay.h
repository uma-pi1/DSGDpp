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
 * Concepts for decay functions.
 */
#ifndef SGD_DECAY_DECAY_H
#define SGD_DECAY_DECAY_H

/** Static decay concept. The decay is computed as a function of the step number
 * (e.g., eps_n = 1/n), but does not have access to data or progress information.
 *
 * \par Methods:
 * <table>
 * <tr><th>Signature</th><th>Description</th></tr>
 * <tr><td>double operator()(unsigned n) const</td><td>Returns the step size to use in the n-th step (0-based)</td></tr>
 * </table>
 *
 * @see AdaptiveDecayConcept
 */
class StaticDecayConcept {
};

/** Adaptive decay concept. An adaptive decay function may monitor the progress of the stochastic
 * optimization to make its choices. It may store local state.
 *
 * \par Methods:
 * <table>
 * <tr><th>Signature</th><th>Description</th></tr>
 * <tr><td>double operator()(FactorizationData& data, double* prevLoss, double* curLoss, rg::Random32& random)</td>
 * <td>Returns the next step size to use. Here, prevLoss and curLoss correspond to the value of the loss
 * at the current and previous iteration, respectively. Note that prevLoss can be set
 * to NULL in the first step.</td></tr>
 * </table>
 *
 * @see StaticDecayConcept
 */
class AdaptiveDecayConcept {
};

/** Adaptive decay concept. An adaptive decay function may monitor the progress of the stochastic
 * optimization to make its choices. It may store local state.
 *
 * \par Methods:
 * <table>
 * <tr><th>Signature</th><th>Description</th></tr>
 * <tr><td>double operator()(DsgdFactorizationData& data, double* prevLoss, double* curLoss, rg::Random32& random)</td>
 * <td>Returns the next step size to use. Here, prevLoss and curLoss correspond to the value of the loss
 * at the current and previous iteration, respectively. Note that prevLoss can be set
 * to NULL in the first step.</td></tr>
 * </table>
 *
 * @see StaticDecayConcept
 */
class DistributedAdaptiveDecayConcept {
};



#endif

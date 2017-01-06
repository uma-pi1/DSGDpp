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
#include <iostream> 
#include <mpi2/mpi2.h> 
#include <mf/mf.h> 
#include "mfdsgd-args.h" 
#include <cmath> 
#include "mfdsgd-generated.h" 
#include "mfdsgd-run.h" 


bool runArgs(Args& args) {
	using namespace mf;
	using namespace mpi2;
	if (!(args.abs)) {
		if (!(!std::isnan(args.truncateArgs[0]) && !std::isnan(args.truncateArgs[1]))) {
			if ((args.updateName.compare("GklData") == 0) && (args.regularizeName.compare("GklModel") == 0) && (args.lossName.compare("Gkl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateGkl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateGkl();
				}
				UpdateGkl updateAbs = update;
				UpdateGkl updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeGkl regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeGkl();
				}
				RegularizeGkl regularizeAbs = regularize;
				RegularizeGkl regularizeTruncate = regularizeAbs;
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				GklLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = GklLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("SlData") == 0) && (args.regularizeName.compare("SlModel") == 0) && (args.lossName.compare("Sl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateSl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateSl();
				}
				UpdateSl updateAbs = update;
				UpdateSl updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeSl regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeSl();
				}
				RegularizeSl regularizeAbs = regularize;
				RegularizeSl regularizeTruncate = regularizeAbs;
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				SlLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = SlLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzsl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateNzsl();
				}
				UpdateNzsl updateAbs = update;
				UpdateNzsl updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeNone regularizeAbs = regularize;
				RegularizeNone regularizeTruncate = regularizeAbs;
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				NzslLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = NzslLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl_L2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl_L2") == 0)) {
				if (args.updateArgs.size()<1 || args.updateArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzslL2 update(UNINITIALIZED);
				if (args.updateArgs.size()==1) {
					update = UpdateNzslL2(args.updateArgs[0]);
				}
				UpdateNzslL2 updateAbs = update;
				UpdateNzslL2 updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeNone regularizeAbs = regularize;
				RegularizeNone regularizeTruncate = regularizeAbs;
				NzslLoss loss1;
				if (args.lossArgs.size()<1 || args.lossArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				L2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==1) {
					loss2 = L2Loss(args.lossArgs[0]);
				}
				SumLoss<NzslLoss, L2Loss> loss = SumLoss<NzslLoss, L2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl_Nzl2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl_Nzl2") == 0)) {
				if (args.updateArgs.size()<1 || args.updateArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzslNzl2 update(UNINITIALIZED);
				if (args.updateArgs.size()==1) {
					update = UpdateNzslNzl2(args.updateArgs[0]);
				}
				UpdateNzslNzl2 updateAbs = update;
				UpdateNzslNzl2 updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeNone regularizeAbs = regularize;
				RegularizeNone regularizeTruncate = regularizeAbs;
				NzslLoss loss1;
				if (args.lossArgs.size()<1 || args.lossArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				Nzl2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==1) {
					loss2 = Nzl2Loss(args.lossArgs[0]);
				}
				SumLoss<NzslLoss, Nzl2Loss> loss = SumLoss<NzslLoss, Nzl2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Biased_Nzsl_Nzl2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Biased_Nzsl_Nzl2") == 0)) {
				if (args.updateArgs.size()<4 || args.updateArgs.size()>4) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateBiasedNzslNzl2 update(UNINITIALIZED);
				if (args.updateArgs.size()==4) {
					update = UpdateBiasedNzslNzl2(args.updateArgs[0], args.updateArgs[1], args.updateArgs[2], args.updateArgs[3]);
				}
				UpdateBiasedNzslNzl2 updateAbs = update;
				UpdateBiasedNzslNzl2 updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeNone regularizeAbs = regularize;
				RegularizeNone regularizeTruncate = regularizeAbs;
				BiasedNzslLoss loss1;
				if (args.lossArgs.size()<4 || args.lossArgs.size()>4) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				BiasedNzl2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==4) {
					loss2 = BiasedNzl2Loss(args.lossArgs[0], args.lossArgs[1], args.lossArgs[2], args.lossArgs[3]);
				}
				SumLoss<BiasedNzslLoss, BiasedNzl2Loss> loss = SumLoss<BiasedNzslLoss, BiasedNzl2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
		}
		if ((!std::isnan(args.truncateArgs[0]) && !std::isnan(args.truncateArgs[1]))) {
			if ((args.updateName.compare("GklData") == 0) && (args.regularizeName.compare("GklModel") == 0) && (args.lossName.compare("Gkl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateGkl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateGkl();
				}
				UpdateGkl updateAbs = update;
				UpdateTruncate<UpdateGkl > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeGkl regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeGkl();
				}
				RegularizeGkl regularizeAbs = regularize;
				RegularizeTruncate<RegularizeGkl > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				GklLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = GklLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("SlData") == 0) && (args.regularizeName.compare("SlModel") == 0) && (args.lossName.compare("Sl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateSl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateSl();
				}
				UpdateSl updateAbs = update;
				UpdateTruncate<UpdateSl > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeSl regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeSl();
				}
				RegularizeSl regularizeAbs = regularize;
				RegularizeTruncate<RegularizeSl > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				SlLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = SlLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzsl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateNzsl();
				}
				UpdateNzsl updateAbs = update;
				UpdateTruncate<UpdateNzsl > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeNone regularizeAbs = regularize;
				RegularizeTruncate<RegularizeNone > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				NzslLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = NzslLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl_L2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl_L2") == 0)) {
				if (args.updateArgs.size()<1 || args.updateArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzslL2 update(UNINITIALIZED);
				if (args.updateArgs.size()==1) {
					update = UpdateNzslL2(args.updateArgs[0]);
				}
				UpdateNzslL2 updateAbs = update;
				UpdateTruncate<UpdateNzslL2 > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeNone regularizeAbs = regularize;
				RegularizeTruncate<RegularizeNone > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				NzslLoss loss1;
				if (args.lossArgs.size()<1 || args.lossArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				L2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==1) {
					loss2 = L2Loss(args.lossArgs[0]);
				}
				SumLoss<NzslLoss, L2Loss> loss = SumLoss<NzslLoss, L2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl_Nzl2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl_Nzl2") == 0)) {
				if (args.updateArgs.size()<1 || args.updateArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzslNzl2 update(UNINITIALIZED);
				if (args.updateArgs.size()==1) {
					update = UpdateNzslNzl2(args.updateArgs[0]);
				}
				UpdateNzslNzl2 updateAbs = update;
				UpdateTruncate<UpdateNzslNzl2 > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeNone regularizeAbs = regularize;
				RegularizeTruncate<RegularizeNone > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				NzslLoss loss1;
				if (args.lossArgs.size()<1 || args.lossArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				Nzl2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==1) {
					loss2 = Nzl2Loss(args.lossArgs[0]);
				}
				SumLoss<NzslLoss, Nzl2Loss> loss = SumLoss<NzslLoss, Nzl2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Biased_Nzsl_Nzl2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Biased_Nzsl_Nzl2") == 0)) {
				if (args.updateArgs.size()<4 || args.updateArgs.size()>4) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateBiasedNzslNzl2 update(UNINITIALIZED);
				if (args.updateArgs.size()==4) {
					update = UpdateBiasedNzslNzl2(args.updateArgs[0], args.updateArgs[1], args.updateArgs[2], args.updateArgs[3]);
				}
				UpdateBiasedNzslNzl2 updateAbs = update;
				UpdateTruncate<UpdateBiasedNzslNzl2 > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeNone regularizeAbs = regularize;
				RegularizeTruncate<RegularizeNone > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				BiasedNzslLoss loss1;
				if (args.lossArgs.size()<4 || args.lossArgs.size()>4) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				BiasedNzl2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==4) {
					loss2 = BiasedNzl2Loss(args.lossArgs[0], args.lossArgs[1], args.lossArgs[2], args.lossArgs[3]);
				}
				SumLoss<BiasedNzslLoss, BiasedNzl2Loss> loss = SumLoss<BiasedNzslLoss, BiasedNzl2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
		}
	}
	if ((args.abs)) {
		if (!(!std::isnan(args.truncateArgs[0]) && !std::isnan(args.truncateArgs[1]))) {
			if ((args.updateName.compare("GklData") == 0) && (args.regularizeName.compare("GklModel") == 0) && (args.lossName.compare("Gkl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateGkl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateGkl();
				}
				UpdateAbs<UpdateGkl> updateAbs(update);
				UpdateAbs<UpdateGkl> updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeGkl regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeGkl();
				}
				RegularizeAbs<RegularizeGkl> regularizeAbs(regularize);
				RegularizeAbs<RegularizeGkl> regularizeTruncate = regularizeAbs;
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				GklLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = GklLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("SlData") == 0) && (args.regularizeName.compare("SlModel") == 0) && (args.lossName.compare("Sl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateSl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateSl();
				}
				UpdateAbs<UpdateSl> updateAbs(update);
				UpdateAbs<UpdateSl> updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeSl regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeSl();
				}
				RegularizeAbs<RegularizeSl> regularizeAbs(regularize);
				RegularizeAbs<RegularizeSl> regularizeTruncate = regularizeAbs;
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				SlLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = SlLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzsl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateNzsl();
				}
				UpdateAbs<UpdateNzsl> updateAbs(update);
				UpdateAbs<UpdateNzsl> updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeAbs<RegularizeNone> regularizeAbs(regularize);
				RegularizeAbs<RegularizeNone> regularizeTruncate = regularizeAbs;
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				NzslLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = NzslLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl_L2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl_L2") == 0)) {
				if (args.updateArgs.size()<1 || args.updateArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzslL2 update(UNINITIALIZED);
				if (args.updateArgs.size()==1) {
					update = UpdateNzslL2(args.updateArgs[0]);
				}
				UpdateAbs<UpdateNzslL2> updateAbs(update);
				UpdateAbs<UpdateNzslL2> updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeAbs<RegularizeNone> regularizeAbs(regularize);
				RegularizeAbs<RegularizeNone> regularizeTruncate = regularizeAbs;
				NzslLoss loss1;
				if (args.lossArgs.size()<1 || args.lossArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				L2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==1) {
					loss2 = L2Loss(args.lossArgs[0]);
				}
				SumLoss<NzslLoss, L2Loss> loss = SumLoss<NzslLoss, L2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl_Nzl2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl_Nzl2") == 0)) {
				if (args.updateArgs.size()<1 || args.updateArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzslNzl2 update(UNINITIALIZED);
				if (args.updateArgs.size()==1) {
					update = UpdateNzslNzl2(args.updateArgs[0]);
				}
				UpdateAbs<UpdateNzslNzl2> updateAbs(update);
				UpdateAbs<UpdateNzslNzl2> updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeAbs<RegularizeNone> regularizeAbs(regularize);
				RegularizeAbs<RegularizeNone> regularizeTruncate = regularizeAbs;
				NzslLoss loss1;
				if (args.lossArgs.size()<1 || args.lossArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				Nzl2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==1) {
					loss2 = Nzl2Loss(args.lossArgs[0]);
				}
				SumLoss<NzslLoss, Nzl2Loss> loss = SumLoss<NzslLoss, Nzl2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Biased_Nzsl_Nzl2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Biased_Nzsl_Nzl2") == 0)) {
				if (args.updateArgs.size()<4 || args.updateArgs.size()>4) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateBiasedNzslNzl2 update(UNINITIALIZED);
				if (args.updateArgs.size()==4) {
					update = UpdateBiasedNzslNzl2(args.updateArgs[0], args.updateArgs[1], args.updateArgs[2], args.updateArgs[3]);
				}
				UpdateAbs<UpdateBiasedNzslNzl2> updateAbs(update);
				UpdateAbs<UpdateBiasedNzslNzl2> updateTruncate = updateAbs;
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeAbs<RegularizeNone> regularizeAbs(regularize);
				RegularizeAbs<RegularizeNone> regularizeTruncate = regularizeAbs;
				BiasedNzslLoss loss1;
				if (args.lossArgs.size()<4 || args.lossArgs.size()>4) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				BiasedNzl2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==4) {
					loss2 = BiasedNzl2Loss(args.lossArgs[0], args.lossArgs[1], args.lossArgs[2], args.lossArgs[3]);
				}
				SumLoss<BiasedNzslLoss, BiasedNzl2Loss> loss = SumLoss<BiasedNzslLoss, BiasedNzl2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
		}
		if ((!std::isnan(args.truncateArgs[0]) && !std::isnan(args.truncateArgs[1]))) {
			if ((args.updateName.compare("GklData") == 0) && (args.regularizeName.compare("GklModel") == 0) && (args.lossName.compare("Gkl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateGkl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateGkl();
				}
				UpdateAbs<UpdateGkl> updateAbs(update);
				UpdateTruncate<UpdateAbs<UpdateGkl> > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeGkl regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeGkl();
				}
				RegularizeAbs<RegularizeGkl> regularizeAbs(regularize);
				RegularizeTruncate<RegularizeAbs<RegularizeGkl> > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				GklLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = GklLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("SlData") == 0) && (args.regularizeName.compare("SlModel") == 0) && (args.lossName.compare("Sl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateSl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateSl();
				}
				UpdateAbs<UpdateSl> updateAbs(update);
				UpdateTruncate<UpdateAbs<UpdateSl> > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeSl regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeSl();
				}
				RegularizeAbs<RegularizeSl> regularizeAbs(regularize);
				RegularizeTruncate<RegularizeAbs<RegularizeSl> > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				SlLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = SlLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl") == 0)) {
				if (args.updateArgs.size()<0 || args.updateArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzsl update(UNINITIALIZED);
				if (args.updateArgs.size()==0) {
					update = UpdateNzsl();
				}
				UpdateAbs<UpdateNzsl> updateAbs(update);
				UpdateTruncate<UpdateAbs<UpdateNzsl> > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeAbs<RegularizeNone> regularizeAbs(regularize);
				RegularizeTruncate<RegularizeAbs<RegularizeNone> > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.lossArgs.size()<0 || args.lossArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				NzslLoss loss(UNINITIALIZED);
				if (args.lossArgs.size()==0) {
					loss = NzslLoss();
				}
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl_L2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl_L2") == 0)) {
				if (args.updateArgs.size()<1 || args.updateArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzslL2 update(UNINITIALIZED);
				if (args.updateArgs.size()==1) {
					update = UpdateNzslL2(args.updateArgs[0]);
				}
				UpdateAbs<UpdateNzslL2> updateAbs(update);
				UpdateTruncate<UpdateAbs<UpdateNzslL2> > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeAbs<RegularizeNone> regularizeAbs(regularize);
				RegularizeTruncate<RegularizeAbs<RegularizeNone> > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				NzslLoss loss1;
				if (args.lossArgs.size()<1 || args.lossArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				L2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==1) {
					loss2 = L2Loss(args.lossArgs[0]);
				}
				SumLoss<NzslLoss, L2Loss> loss = SumLoss<NzslLoss, L2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Nzsl_Nzl2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Nzsl_Nzl2") == 0)) {
				if (args.updateArgs.size()<1 || args.updateArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateNzslNzl2 update(UNINITIALIZED);
				if (args.updateArgs.size()==1) {
					update = UpdateNzslNzl2(args.updateArgs[0]);
				}
				UpdateAbs<UpdateNzslNzl2> updateAbs(update);
				UpdateTruncate<UpdateAbs<UpdateNzslNzl2> > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeAbs<RegularizeNone> regularizeAbs(regularize);
				RegularizeTruncate<RegularizeAbs<RegularizeNone> > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				NzslLoss loss1;
				if (args.lossArgs.size()<1 || args.lossArgs.size()>1) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				Nzl2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==1) {
					loss2 = Nzl2Loss(args.lossArgs[0]);
				}
				SumLoss<NzslLoss, Nzl2Loss> loss = SumLoss<NzslLoss, Nzl2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
			if ((args.updateName.compare("Biased_Nzsl_Nzl2") == 0) && (args.regularizeName.compare("None") == 0) && (args.lossName.compare("Biased_Nzsl_Nzl2") == 0)) {
				if (args.updateArgs.size()<4 || args.updateArgs.size()>4) {
					std::cout << "Invalid number of arguments in " << args.updateString << std::endl;
					return false;
				}
				UpdateBiasedNzslNzl2 update(UNINITIALIZED);
				if (args.updateArgs.size()==4) {
					update = UpdateBiasedNzslNzl2(args.updateArgs[0], args.updateArgs[1], args.updateArgs[2], args.updateArgs[3]);
				}
				UpdateAbs<UpdateBiasedNzslNzl2> updateAbs(update);
				UpdateTruncate<UpdateAbs<UpdateBiasedNzslNzl2> > updateTruncate(updateAbs, args.truncateArgs[0], args.truncateArgs[1]);
				if (args.regularizeArgs.size()<0 || args.regularizeArgs.size()>0) {
					std::cout << "Invalid number of arguments in " << args.regularizeString << std::endl;
					return false;
				}
				RegularizeNone regularize(UNINITIALIZED);
				if (args.regularizeArgs.size()==0) {
					regularize = RegularizeNone();
				}
				RegularizeAbs<RegularizeNone> regularizeAbs(regularize);
				RegularizeTruncate<RegularizeAbs<RegularizeNone> > regularizeTruncate(regularizeAbs, args.truncateArgs[0], args.truncateArgs[1]);
				BiasedNzslLoss loss1;
				if (args.lossArgs.size()<4 || args.lossArgs.size()>4) {
					std::cout << "Invalid number of arguments in " << args.lossString << std::endl;
					return false;
				}
				BiasedNzl2Loss loss2(UNINITIALIZED);
				if (args.lossArgs.size()==4) {
					loss2 = BiasedNzl2Loss(args.lossArgs[0], args.lossArgs[1], args.lossArgs[2], args.lossArgs[3]);
				}
				SumLoss<BiasedNzslLoss, BiasedNzl2Loss> loss = SumLoss<BiasedNzslLoss, BiasedNzl2Loss>(loss1, loss2);
				runDsgd(args, updateTruncate, regularizeTruncate, loss);
				return true;
			}
		}
	}
	std::cerr << "Invalid combination of update, regularize, and loss arguments" << std::endl;
	std::cout << "Valid combinations are:" << std::endl;
	std::cout << "	GklData / GklModel / Gkl" << std::endl;
	std::cout << "	SlData / SlModel / Sl" << std::endl;
	std::cout << "	Nzsl / None / Nzsl" << std::endl;
	std::cout << "	Nzsl_L2 / None / Nzsl_L2" << std::endl;
	std::cout << "	Nzsl_Nzl2 / None / Nzsl_Nzl2" << std::endl;
	std::cout << "	Biased_Nzsl_Nzl2 / None / Biased_Nzsl_Nzl2" << std::endl;
	return false;
}


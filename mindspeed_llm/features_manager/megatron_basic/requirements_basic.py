# Copyright (c) 2025, Huawei Technologies Co., Ltd.  All rights reserved.
import sys
from argparse import ArgumentParser

import torch
from mindspeed.features_manager.megatron_basic.requirements_basic import RequirementsBasicFeature as MindspeedRequirementsBasicFeature


class RequirementsBasicFeature(MindspeedRequirementsBasicFeature):

    def register_args(self, parser: ArgumentParser):
        super().register_args(parser)
        group = parser.add_argument_group(title=self.feature_name)
        group.add_argument('--o2-optimizer', action='store_true',
                            help='use bf16 exponential moving average to greatly save up memory.')
        group.add_argument('--o2-gradient', action='store_true',
                            help='use bf16 gradient accumulation to greatly save up memory.')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'zeta_moon')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'muon')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'adamuon')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'soap')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'zeta_sophia')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'sophia')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'zetas')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'sophiamuon')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'sophiamuonflatness')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'zetar')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'signmuon')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'signmuonfeature')
        self.add_parser_argument_choices_value(parser, "--optimizer-selection", 'vpmc')
        group.add_argument('--zeta-momentum', type=float, default=0.95,
                            help='momentum coefficient for ZetaMoon optimizer')
        group.add_argument('--zeta-ns-steps', type=int, default=5,
                            help='Newton-Schulz iteration steps for ZetaMoon optimizer')
        group.add_argument('--zeta-trust-region-threshold', type=float, default=0.1,
                            help='trust region gating threshold for ZetaMoon optimizer')
        group.add_argument('--sophia-hessian-interval', type=int, default=10,
                            help='interval (k) for SophiaG Hessian updates')
    
    def register_patches(self, patch_manager, args):
        super().register_patches(patch_manager, args)
        self.version_patch(patch_manager, args)

    def pre_register_patches(self, patch_manager, args):
        super().pre_register_patches(patch_manager, args)
        self.load_checkpoint_patch(patch_manager, args)
        
    def optimizer_selection(self, pm, args):
        from mindspeed.core.optimizer.adamw import FusedTorchAdamW, AdamW
        if args.o2_optimizer:
            # O2 optimizer
            from mindspeed_llm.tasks.models.common.adamw import O2AdamW
            pm.register_patch('apex.optimizers.FusedAdam', 
                               O2AdamW, create_dummy=True)
            
        else:
            if args.optimizer_selection == 'fused_torch_adamw':
                pm.register_patch('apex.optimizers.FusedAdam', 
                                   FusedTorchAdamW, create_dummy=True)
            elif args.optimizer_selection == 'fused_adamw':
                pm.register_patch('apex.optimizers.FusedAdam', 
                                   AdamW, create_dummy=True)
            elif args.optimizer_selection == 'zeta_moon':
                from mindspeed_llm.tasks.models.common.zeta_moon import ZetaMoon
                pm.register_patch('apex.optimizers.FusedAdam',
                                   ZetaMoon, create_dummy=True)
            elif args.optimizer_selection == 'muon':
                from mindspeed_llm.tasks.models.common.muon import Muon
                pm.register_patch('apex.optimizers.FusedAdam',
                                   Muon, create_dummy=True)
            elif args.optimizer_selection == 'adamuon':
                from mindspeed_llm.tasks.models.common.adamuon import AdaMuon
                pm.register_patch('apex.optimizers.FusedAdam',
                                   AdaMuon, create_dummy=True)
            elif args.optimizer_selection == 'soap':
                from mindspeed_llm.tasks.models.common.soap import SOAP
                pm.register_patch('apex.optimizers.FusedAdam',
                                   SOAP, create_dummy=True)
            elif args.optimizer_selection == 'zeta_sophia':
                from mindspeed_llm.tasks.models.common.zeta_sophia import ZetaSophia
                pm.register_patch('apex.optimizers.FusedAdam',
                                   ZetaSophia, create_dummy=True)
            elif args.optimizer_selection == 'sophia':
                from mindspeed_llm.tasks.models.common.sophia import SophiaG
                pm.register_patch('apex.optimizers.FusedAdam',
                                   SophiaG, create_dummy=True)
            elif args.optimizer_selection == 'zetas':
                from mindspeed_llm.tasks.models.common.zetas import ZetaS
                pm.register_patch('apex.optimizers.FusedAdam',
                                   ZetaS, create_dummy=True)
            elif args.optimizer_selection == 'sophiamuon':
                from mindspeed_llm.tasks.models.common.sophiamuon import SophiaMuon
                pm.register_patch('apex.optimizers.FusedAdam',
                                   SophiaMuon, create_dummy=True)
            elif args.optimizer_selection == 'sophiamuonflatness':
                from mindspeed_llm.tasks.models.common.sophiamuon_flatness import SophiaMuon                         
                pm.register_patch('apex.optimizers.FusedAdam',
                                   SophiaMuon, create_dummy=True)
            elif args.optimizer_selection == 'zetar':
                from mindspeed_llm.tasks.models.common.zetar import ZetaR                         
                pm.register_patch('apex.optimizers.FusedAdam',
                                   ZetaR, create_dummy=True)
            elif args.optimizer_selection == 'signmuon':
                from mindspeed_llm.tasks.models.common.signmuon import SignMuon                         
                pm.register_patch('apex.optimizers.FusedAdam',
                                   SignMuon, create_dummy=True)
            elif args.optimizer_selection == 'signmuonfeature':
                from mindspeed_llm.tasks.models.common.signmuon_feature import SignMuon                         
                pm.register_patch('apex.optimizers.FusedAdam',
                                   SignMuon, create_dummy=True)
            elif args.optimizer_selection == 'vpmc':
                from mindspeed_llm.tasks.models.common.vpmc import VPMC                         
                pm.register_patch('apex.optimizers.FusedAdam',
                                   VPMC, create_dummy=True)
            pm.register_patch('apex.optimizers.FusedSGD', 
                               torch.optim.SGD, create_dummy=True)

    def version_patch(self, pm, args):
        from mindspeed_llm.tasks.megatron_basic.requirements_basic import version_wrapper
        pm.register_patch('importlib.metadata.version', version_wrapper)

    def load_checkpoint_patch(self, pm, args):
        if hasattr(args, 'lora_target_modules') and args.lora_target_modules:
            from mindspeed_llm.tasks.megatron_basic.requirements_basic import _load_from_state_dict_wrapper
            pm.register_patch('torch.nn.Module._load_from_state_dict', _load_from_state_dict_wrapper)

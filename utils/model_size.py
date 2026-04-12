
import torch

# Load the state dict
state_dict = torch.load("Experiments/SIM+_n5_channelFusion.pt", weights_only=True)
state_dict = torch.load("Experiments/SIM+_n5_lateFusion_onlyCenter.pt", weights_only=True)


# Print parameters per module from state dict
def print_params_from_state_dict(state_dict):
    # Group by module prefix
    module_params = {}
    
    for name, param in state_dict.items():
        # print(name)
        # Extract the top-level module name
        if 'backbone' in name:
            module_name = 'backbone'
        elif 'rpn' in name:
            module_name = 'rpn'
        elif 'roi_heads' in name:
            # if 'mask' in name:
            #     module_name = 'roi_heads.mask_predictor'
            # elif 'box' in name:
            #     module_name = 'roi_heads.box_predictor'
            # else:
            module_name = 'roi_heads'
        elif 'transform' in name:
            module_name = 'transform'
        else:
            module_name = 'other'
        
        if module_name not in module_params:
            module_params[module_name] = 0
        module_params[module_name] += param.numel()
    
    # Print results
    print("\nParameters per module:")
    print("-" * 60)
    for module, count in sorted(module_params.items()):
        print(f"{module:30s} | {count:>12,d} parameters")
    
    total = sum(module_params.values())
    print("-" * 60)
    print(f"{'Total':30s} | {total:>12,d} parameters")

print_params_from_state_dict(state_dict)
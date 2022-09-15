

def get_model(save_manager):
    name = save_manager.config.Model.NAME

    if name == 'RITM_SE_HRNet32':
        from model.iterativeRefinementModels.RITM_SE_HRNet32 import RITM as Model
    else:
        save_manager.write_log('ERROR: NOT SPECIFIED MODEL NAME: {}'.format(name))
        raise NotImplemented

    model = Model(save_manager.config)
    return model
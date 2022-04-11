import os 

def checkpoints():
    modes = ['teacher', 'experts']

    for mode in modes:
        folder_source = 'checkpoints/{}'.format(mode)
        folder_target = 'checkpoints'
        names = [name for name in os.listdir(folder_source)]
        for name in names:
            source = os.path.join(folder_source, name)
            target = os.path.join(folder_target, name[:-4] + '-{}.pth'.format(mode))
            os.system(f'cp {source} {target}')
        
def configs():
    
    folder_source = 'configs/experts'
    folder_target = 'configs'
    names = [name for name in os.listdir(folder_source)]
    for name in names:
        source = os.path.join(folder_source, name)
        target = os.path.join(folder_target, name)
        os.system(f'cp {source} {target}')

if __name__ == '__main__':
    # checkpoints()
    configs()
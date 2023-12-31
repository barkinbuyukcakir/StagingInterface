import multiprocessing as mp
import subprocess
from aevit_cv2 import CrossValidator
import argparse

def process_test_results():
    pass

def cross_validate(tooth,gpu,epochs,clahe,randomaffine,patch_size,silent):
    print(f"Cross-validating tooth {tooth} on GPU{gpu} PatchSize{patch_size}| {'CL' if clahe else ''} {'RA' if randomaffine else ''}")
    
    CrossValidator(
        tooth=tooth,
        gpu = gpu,
        vit_epochs=epochs,
        ae_epochs=80,
        clahe=clahe,
        randomaffine=randomaffine,
        patch_size=patch_size,
        silent=silent,
        svd=False,
        layers=12,
        heads=6,
        modelType="AE+ViT"
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train",action="store_true")
    parser.add_argument("--test",action="store_true")
    args = parser.parse_args()
    
    procs = []    
    teeth = [31,33,34,37,38]
    patch_sizes = [16]
    aug_clahe = (False,)
    aug_ra = (False,)
    gpu_track = 0
    gpus = [0,1,2,3,4]
    ct = 0
    if args.train:
        for ps in patch_sizes:
            for tooth in teeth:
                for clahe in aug_clahe:
                    for epochs in [40]:
                        for ra in aug_ra:
                            # if gpu_track==3:
                            #     gpu_track+=1
                            #     continue
                            args_cv = (tooth,gpus[gpu_track],epochs,clahe,ra,ps,True,)
                            ct+=1
                            p = mp.Process(target = cross_validate,args=args_cv)
                            limit = 0 if ps == 32 else 0
                            if ct>limit:
                                gpu_track+=1
                                ct=0
                            procs.append(p)            
                            p.start()

                            if gpu_track>len(gpus)-1:
                                print("Waiting for free gpus")
                                gpu_track=0
                                for proc in procs:
                                    proc.join()

        for p in procs:
            p.join()
    
    
    if args.test:
        print("Multiprocess testing is not implemented. Use cv_test.py.")

print("Complete.")
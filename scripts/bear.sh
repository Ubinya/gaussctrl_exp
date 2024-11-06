ns-train splatfacto --output-dir unedited_models --experiment-name bear --viewer.quit-on-train-completion True nerfstudio-data --data data/bear

ns-train gaussctrl --load-checkpoint unedited_models/bear/splatfacto/2024-07-10_170906/nerfstudio_models/step-000029999.ckpt --experiment-name bear --output-dir outputs --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a polar bear in the forest" --pipeline.reverse_prompt "a photo of a bear statue in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'bear' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/bear/splatfacto/2024-07-10_170906/nerfstudio_models/step-000029999.ckpt --experiment-name bear --output-dir outputs --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a grizzly bear in the forest" --pipeline.reverse_prompt "a photo of a bear statue in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'bear' --viewer.quit-on-train-completion True 

ns-train gaussctrl --load-checkpoint unedited_models/bear/splatfacto/2024-07-10_170906/nerfstudio_models/step-000029999.ckpt --experiment-name bear --output-dir outputs --pipeline.datamanager.data data/bear --pipeline.edit_prompt "a photo of a golden bear statue in the forest" --pipeline.reverse_prompt "a photo of a bear statue in the forest" --pipeline.guidance_scale 5 --pipeline.chunk_size 3 --pipeline.langsam_obj 'bear' --viewer.quit-on-train-completion True 
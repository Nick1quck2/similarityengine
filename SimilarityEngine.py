import fitz  # PyMuPDF
import shutil
import re 
import json
import os
import tensorflow
from keras.applications.resnet50 import ResNet50
from keras.layers import GlobalMaxPooling2D
from keras.applications.resnet50 import preprocess_input
import cv2

from numpy.linalg import norm


import pandas as pd 
import os 
from thefuzz import fuzz,process
from flask import  Flask, jsonify, request
import base64
import torch
import clip
from annoy import AnnoyIndex
from PIL import Image
from transformers import CLIPTokenizerFast , CLIPProcessor  , CLIPModel
import torch






class DataOrganize:
    def __init__(self,folder_path) -> None:
        self.folder_path = folder_path
        self.data={}
        #1. button store_split
       
    
    def find_pdfs( self,  folder_path):
        pdf_files = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pdf") or file.endswith(".PDF") :
                    pdf_files.append(os.path.join(root, file))
        return pdf_files

    def pdf_to_image(self,  pdf_path, dir   ):
        # Open the PDF\
        pdf_document = fitz.open(pdf_path)
        length= len(pdf_document)
        pdfname = pdf_path.replace(".pdf",'').replace(".PDF",'')
    
        pdfname = pdfname [::-1]
        string_store= ''
        for value  in pdfname:
            if value =="/" or value =="\\":
                break
            string_store+= value 
        string_store=string_store[::-1]
        for  number  in range(length):
            first_page = pdf_document[number ]


            
            savepath = os.path.join( dir , f"{string_store}_name_{number}.png")
            
        
            
            pix = first_page.get_pixmap()
            
        
            
            pix.save(savepath)
            
        
        pdf_document.close()
        print(f'{pdf_path} pdf to jpg complete')

    def extract_text_and_images_from_pdf(self , pdf_path, page_number , dir , name  ):
        
        text_and_images = {"text": "", "images": []}
        
        doc = fitz.open(pdf_path)
        page = doc.load_page(page_number)
        
        # Extract text
        text = page.get_text()
        text_and_images["text"] = text
        
        # Extract images
        image_list = page.get_images(full=True)
        name1 = name [::-1]
        string = ''
        for index in name1:
            if index=="/" or index=="\\":
                break
            string+= index 
        string=string[::-1]

        for img_index, img_info in enumerate(image_list):
            xref = img_info[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image_ext = base_image["ext"]
            savepath = os.path.join( dir ,f"{string}_image_{page_number}_{img_index}.{image_ext}"   )
            image_name = savepath
            with open(image_name, "wb") as f:
                f.write(image_bytes) ##change 
            text_and_images["images"].append(image_name)
        if "Trade Marks Journal No:" in text_and_images['text']:
            value= text_and_images['text']
            List= value.split("\n")
            
            first = ''
            first_=1
            start = 0 
            yes = 0 
            for index in range(len(List)):
                if "Trade Marks Journal No:" in List[index]: 
                    start =1 
                elif start:
                    if len((re.findall("\d{2}/\d{2}/\d{4}",List[index])))>=1:
                        break


                    if "Priority claimed" in List[index] :
                        break


                    elif "International Registration No." in List[index]:
                        break
                    elif List[index].count("/")==2:
                        first_= 0 
                        if yes==0:
                            yes = 1
                        else:
                            break
                    if first_:
                        first+= List[index] 
            
        
            self.data[str(string)+"_"+str(page_number)] = {"first": first}

    def store_and_split(self):
        print("Storing data  and Spliting data start  ")
        
        folder = self.folder_path
        all_pdfs = self.find_pdfs(folder)
        
        all_images_path = os.path.join(folder,"images")
        
        cut_images_path = os.path.join(folder,"cutimages" )
        jsonpath = os.path.join(folder,"wordsdata.json")
        if os.path.exists(all_images_path):
            shutil.rmtree(all_images_path)
        if os.path.exists(cut_images_path):
            shutil.rmtree(cut_images_path)
        if os.path.exists(jsonpath):
            os.remove(jsonpath)
        os.mkdir( all_images_path)
        os.mkdir(cut_images_path)


        for pdf in all_pdfs:
            name = pdf.replace(".pdf",'').replace(".PDF",'')
            
            self.pdf_to_image (pdf,all_images_path  )
            
            

            length = fitz.open(pdf)
            for  pdf_index  in range(len(length)):
                self.extract_text_and_images_from_pdf(   pdf , pdf_index , cut_images_path , name    )


        json_data = json.dumps(self.data)
        with open(jsonpath, "w") as json_file:
            json_file.write(json_data)
        
        image_paths = self.list_image_paths( cut_images_path)
        for_delete = []
        for path in image_paths:
            if str(path).endswith("_0.jpg") or str(path).endswith("_0.jpeg") or str(path).endswith("_0.png"):

                pass
                
            else:
                for_delete.append(path)
        for image  in for_delete:
            os.remove( image )

        print("Storing data  and Spliting data Complete  ")
        maded = [all_images_path,cut_images_path, jsonpath]
        print("following files are maded ! ----")

        for files_maded in maded:
            print( files_maded  )

            


        
        return maded
            
   

    def list_image_paths(self,folder_path):
        image_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp']  # Add more extensions if needed
        image_paths = []

        for root, dirs, files in os.walk(folder_path):
            for file in files:
                _, extension = os.path.splitext(file)
                if extension.lower() in image_extensions:
                    image_paths.append(os.path.join(root, file))

        return image_paths

class ModelSelection_Training:
    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path

        
        #1 button training
        
        
    


    def training(self  ):
        
        
        paths = os.path.join(self.folder_path , "cutimages")
        device = "cuda" if torch.cuda.is_available() else "cpu"


        model, preprocess = clip.load("ViT-L/14", device=device)

        annoy_index=AnnoyIndex(768,'euclidean')
        counting = 0 

        for filename in os.listdir(paths):
            if filename.endswith(".png") or filename.endswith(".jpg")or filename.endswith(".jpeg"):
                image2_path = os.path.join(paths, filename)
                image2 = preprocess(Image.open(image2_path)).unsqueeze(0).to(device)


                
                # Encode image2
                image2_embedding = model.encode_image(image2)
                annoy_index.add_item(counting ,image2_embedding[0])
                counting+=1 
        

        

        annoy_index.build(10)
        save_path = os.path.join( self.folder_path ,'logo_imageclip_L14.ann' )

        annoy_index.save(save_path)
        print("Images training done !")

        
        jsonpath = os.path.join(self.folder_path , "wordsdata.json")

        df = pd.read_json(  jsonpath )
        df1= df.T.copy()
        df1.dropna(subset=['first'], inplace = True ) 
        df1 = df1[df1['first']!="  "].copy()
        df1= df1[ df1['first']!=""].copy()
        df1= df1[ df1['first']!=" "].copy()
        data=(list(df1['first']))


        data = list(map(  lambda x: str(x).lower() , data ))

       
        parent= os.path.dirname(paths)

        device =  "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu" 




        model_id = "openai/clip-vit-base-patch32"
        
        model = CLIPModel.from_pretrained(model_id).to(device)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)
        annoy_index=AnnoyIndex(512,'euclidean')
        
        counting =0

        for value  in data:
            

            inputs = tokenizer(value , return_tensors = "pt")


            text_emb = model.get_text_features(**inputs)
            annoy_index.add_item(counting ,text_emb[0])


            # print(text_emb[0])
        
            counting+=1 
        

        annoy_index.build(10)
        save_path = os.path.join( self.folder_path ,'logo_wordsclip.ann' )

        annoy_index.save(save_path)
        print("Words training done !")




class Inference:
    def __init__(self, folder_path) -> None:
        self.folder_path = folder_path
    
    def get_similarimages(self, image_path , num = 5):

        try:
            path = os.path.join( self.folder_path , 'cutimages')
            images=os.listdir(path)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, preprocess = clip.load("ViT-L/14", device=device)
            annoy_index=AnnoyIndex(768,'euclidean')

            annoy_index.load( os.path.join(self.folder_path  , 'logo_imageclip_L14.ann'))        
            image2 = preprocess(Image.open(image_path)).unsqueeze(0).to(device)
            image2_embedding = model.encode_image(image2)
            nns=annoy_index.get_nns_by_vector(image2_embedding[0], num)
            pages_path_store = []
            images_path_store = []


            for j in range(num):
                searched_image  = (os.path.join(path,  images[nns[j]]))
                imagename= r''


                for value  in str ( searched_image) :
                    imagename+= value 
           
                


                if imagename.endswith(".jpeg"):
                    pagename =  imagename.replace("cut",'').replace("_0.jpeg",'.png').replace("_image",'_name')

                    
                elif imagename.endswith(".jpg"):
                    pagename =  imagename.replace("cut",'').replace("_0.jpg",'.png').replace("_image",'_name')
                
                elif imagename.endswith(".png"):
                    pagename =  imagename.replace("cut",'').replace("_0.png",'.png').replace("_image",'_name')
                    

                

                pages_path_store.append(pagename)
                images_path_store.append(imagename)

            return images_path_store,pages_path_store
        except Exception as e :
            jsonify({'error in get similarimages ': str(e)})
    

        





            


class WordsSmly:
    def __init__(self, folder_path) -> None:
        self.folder_path  = folder_path
        self.jsonload()
    
    def jsonload(self ):
        jsonpath = os.path.join(self.folder_path , "wordsdata.json")

        df = pd.read_json(  jsonpath )
        df1= df.T.copy()
        df1.dropna(subset=['first'], inplace = True ) 
        df1 = df1[df1['first']!="  "].copy()
        df1= df1[ df1['first']!=""].copy()
        df1= df1[ df1['first']!=" "].copy()
        self.data=(list(df1['first']))
        self.data = list(map(  lambda x: str(x).lower() , self.data ))
        self.page_data=(list(df1.index))
        

       

       

    def get_similar_words(self , input_word ,num=5 ) :
        input_word= input_word.lower()
        device =  "cuda" if torch.cuda.is_available() else \
         "mps" if torch.backends.mps.is_available() else "cpu" 




        model_id = "openai/clip-vit-base-patch32"

        model = CLIPModel.from_pretrained(model_id).to(device)
        tokenizer = CLIPTokenizerFast.from_pretrained(model_id)
        processor = CLIPProcessor.from_pretrained(model_id)


        

        annoy_index=AnnoyIndex(512,'euclidean')
        path = os.path.join(self.folder_path , 'logo_wordsclip.ann' )
        annoy_index.load(path)

        inputs = tokenizer(input_word, return_tensors = "pt")
        text_emb = model.get_text_features(**inputs)
        nns=annoy_index.get_nns_by_vector(text_emb[0], num )

     
        

       
        words_store = []
        page_images_store = []

        for word in range(len(nns)):
                words_store.append(self.data[word]) 


                
                page_images_store.append(  os.path.join(   os.path.join(self.folder_path,"images") , self.page_data[word].replace("_","_name_")+".png"  ))

                 
        return words_store , page_images_store
            

        
#--------------------------------------flask
app = Flask(__name__)



@app.route('/Training',methods=['POST'])
def Training_model():
    try:
        data = request.json
        folder_path = data.get('folder_path')

        obj_organize  =DataOrganize(folder_path)
        obj_organize.store_and_split()

        obj_training = ModelSelection_Training(folder_path)
        obj_training.training()

        return jsonify( {'message': "Training completed" })

    except Exception as e:
        return jsonify( {'message': "error --"+str(e) })



@app.route('/GetImages_Similar/',methods=['POST'])
def ImageSimilarity():
    try:
        data = request.json
        folder_path = data.get('folder_path')
        image_path  = data.get('checkimage_path')
        image_data = base64.b64decode(image_path)


        # Write the image data to a file
        with open('a.png', 'wb') as f:
            f.write(image_data)

        if os.path.exists(folder_path):
            obj = Inference(folder_path)
            result = obj.get_similarimages('a.png' )

            return jsonify({'similarities' :  str(result)  }) 
        # obj = Inference(folder_path)
        # checkimage_path = str(data.get('checkimage_path'))



    
    except Exception as e:
        return jsonify( {'message': "error --" +str(e)})


@app.route('/GetNames_Similar',methods=['POST'])
def WordSimilarity():
    data = request.json
    folder_path = data.get('folder_path')
    
    input_text = data.get('input_text')


    obj = WordsSmly(folder_path)
    result = obj.get_similar_words(input_text)
    return jsonify({"msg":  str(result)  })
    




if __name__ == '__main__': 
    app.run()




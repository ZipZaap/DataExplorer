from darwin.client import Client
# import darwin.importer as importer
# from darwin.importer import get_importer

from slugify import slugify
import glob

from configs import CONF


class DarwinDataset():
    def __init__(self):
        client = Client.from_api_key(CONF.API_KEY)
        try:
            self.dataset = client.get_remote_dataset(f'{slugify(CONF.TEAM)}/{slugify(CONF.DSET)}')
            print(f'Available releases for dataset {self.dataset.name}:')
            if self.dataset.get_releases():
                for release in self.dataset.get_releases():
                    print(release.identifier)   
            else:
                print('...')
        except:
            self.dataset = client.create_dataset(CONF.DSET)

    def upload_img(self):
        img_list = glob.glob(f'{CONF.SEG_IMG_DIR}/*.png')
        self.dataset.push(img_list)

    # def upload_lbl(self):
    #     # label_list = glob.glob(os.path.join('../mro/160x160/', 'labels/V7/dwin-v1.0', '*.json'))
    #     # parser = get_importer('darwin')
    #     # importer.import_annotations(dataset, parser, label_list, append=False)
    #     pass

    # def download_release(self, name):
    #     try:
    #         release = self.dataset.get_release(name)
    #         self.dataset.pull(release=release, only_annotations=True, use_folders=True)
    #     except:
    #         print(f"Dataset release {name} not found")

# # Create release, pull annotations & keep folder structure
# dataset.export('release_name')
# dataset.pull(release=release, only_annotations=True, use_folders=True)
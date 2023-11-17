from torch.utils.data import DataLoader
import load_data


def main():
    monet_dataset = load_data.load_monet()
    photo_dataset = load_data.load_photo()



if __name__ == '__main__':
    main()

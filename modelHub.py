from kgen2.inputs import *

class hub():

    def __init__(self, model, encoder_train, decoder_train, train_y, representation_model=None):
        self.training_model = model
        self.reps = representation_model
        self.encoder_train_x, self.decoder_train_x, self.y = encoder_train, decoder_train, train_y
        super(hub, self).__init__()

    def validate(self, encX, decX, y):
        acc=[]
        loss=[]
        for i, x in enumerate(encX):
            los, ac = self.training_model.evaluate(x+[decX[i]], y[i], verbose=0)
            acc.append(ac)
            loss.append(los)
        print('@acc: {:.4f} | @los: {:.4f}'.format(sum(acc)/len(acc), sum(loss)/len(loss)))
        return acc, loss

    def train(self, epochs=10, validation_data=[], cutoff=.9, return_stats=False):
        loss = []
        acc = []
        for ep in range(epochs):
            print('\nEpoch: {}/{}'.format(ep+1, epochs))
            for i, x in enumerate(self.encoder_train_x):
                self.training_model.train_on_batch(x+[self.decoder_train_x[i]], self.y[i])

            if bool(validation_data):
                ac, los = self.validate(validation_data[0], validation_data[1], validation_data[2])
                acc.append(ac)
                loss.append(los)

            else:
                ac, los = self.validate(self.encoder_train_x, self.decoder_train_x, self.y)
                acc.append(ac)
                loss.append(los)

            print('============] [============')

            if sum(acc[-1])/len(acc[-1]) >= cutoff:
                break

        if return_stats:
            return acc, loss

    def return_representations(self, encX, decX, output_shape=(-1)):
        return [np.array(self.reps.predict(x+[decX[i]])).reshape(output_shape) for i, x in enumerate(encX)]

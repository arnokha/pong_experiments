class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()

        self.fc1 = nn.Linear(im_dm, 800)
        #self.fc1 = nn.Linear(80*80, 800)
        self.fc21 = nn.Linear(800, z_dim)
        self.fc22 = nn.Linear(800, z_dim)
        self.fc3 = nn.Linear(z_dim, 800)
        self.fc4 = nn.Linear(800, im_dm)
        #self.fc4 = nn.Linear(800, 80*80)

    def encode(self, x):
        h1 = torch.sigmoid(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = torch.sigmoid(self.fc3(z))
        return torch.relu(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, im_dm))
        #mu, logvar = self.encode(x.view(-1, 80*80))
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
    
    
//
//  GameBoardViewController.swift
//  ios_ws
//
//  Created by user1 on 2/4/2020.
//  Copyright © 2020 COMP7506. All rights reserved.
//

import UIKit

class GameBoardViewController: UIViewController {
    override func viewDidLoad() {
        super.viewDidLoad()
        self.navigationItem.leftBarButtonItem = UIBarButtonItem(title: "Back", style: UIBarButtonItem.Style.plain, target: self, action: #selector(GameBoardViewController.backButtonItemToDismissModal))
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    func setNavTitle(newUserName: String) {
        self.navigationItem.title = newUserName + "'s Card 24 game"
    }
    
    @objc func backButtonItemToDismissModal () {
        self.dismiss(animated: true)
    }
}

//
//  ViewController.swift
//  ios_ws
//
//  Created by user1 on 2/4/2020.
//  Copyright © 2020 COMP7506. All rights reserved.
//

import UIKit

class ViewController: UIViewController {

    @IBOutlet weak var userNameTF: UITextField!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        // Do any additional setup after loading the view.
    }
    
    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
    }
    
    override func prepare(for segue: UIStoryboardSegue, sender: Any?) {
        if segue.identifier == "loginToGameBoardSeg" {
            let nav = segue.destination as! UINavigationController
            let vc = nav.topViewController as! GameBoardViewController
            vc.setNavTitle(newUserName: userNameTF.text!)
        }
    }
}

